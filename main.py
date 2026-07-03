import cv2
import numpy as np
import mss
import torch
from ultralytics import YOLO
import time
import random
import threading
import ctypes
import os
import sys
import logging
import gc
import math
import atexit
from flask import Flask, render_template_string, request, jsonify

# ==========================================
# 0. 系統環境防禦與全域低耗優化
# ==========================================
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# [穩定] 鎖定 Windows 核心高精度時鐘 (1ms 級精準度)
try:
    ctypes.windll.winmm.timeBeginPeriod(1)
    def cleanup_timer():
        ctypes.windll.winmm.timeEndPeriod(1)
    atexit.register(cleanup_timer)
except Exception:
    pass

def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_path()
sys.path.append(BASE_DIR)

# ==========================================
# 1. 系統全域參數與安全執行緒配置中心
# ==========================================
def discover_pt_models():
    """ [兼容] 支援 TensorRT (.engine) 終極加速，優先載入 """
    try:
        files = [f for f in os.listdir(BASE_DIR) if f.endswith(('.engine', '.pt', '_openvino_model'))]
        # 過濾掉自動產生的 OpenVINO 資料夾後綴
        files = [f for f in files if not f.endswith('_openvino_model')]
        if not files: return ["yolo11n.pt"]
        
        if "best.engine" in files:
            files.insert(0, files.pop(files.index("best.engine")))
        elif "best.pt" in files:
            files.insert(0, files.pop(files.index("best.pt")))
        else:
            files = sorted(files)
        return files
    except Exception:
        return ["best.pt"]

class SystemConfig:
    def __init__(self):
        self.lock = threading.Lock()
        self.box_size = 400
        self.smooth = 0.35
        self.target_part = 1  # 0=頭, 1=胸, 2=腹, 3=手, 4=腳
        self.human_noise = 0.8  
        self.alpha = 0.30
        self.deadzone = 2.0
        self.fov = 100
        
        self.aimbot_enabled = False   
        self.rage_mode = False        
        self.continuous_aim = False   
        self.sniper_mode = False      
        
        # 💡 [新增] 支援可選的硬體運算後端: "cuda", "cpu", "npu"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.show_window = False      
        self.max_fps = 144 
        
        self.recoil_enabled = False   
        self.recoil_strength = 1.2    
        
        self.available_models = discover_pt_models()
        self.model_file = self.available_models[0]
        
        self.kmbox_ip = "192.168.2.188"
        self.kmbox_port = "1234"
        self.kmbox_uuid = "88888888"

        # 💡 [整合] 動態熱鍵配置中心
        self.hotkey_code = 0x02  # 預設：滑鼠右鍵
        self.key_map = {
            "right_mouse": 0x02,
            "left_mouse": 0x01,
            "mouse_x1": 0x05,
            "mouse_x2": 0x06,
            "shift": 0x10,
            "alt": 0x12
        }

cfg = SystemConfig()

# ==========================================
# 2. 三角洲行動專屬：動態矩陣壓槍系統 
# ==========================================
class DeltaForceRecoilSystem:
    def __init__(self):
        self.shot_tick = 0  

    def get_offset(self, strength, is_ads):
        self.shot_tick += 1
        if self.shot_tick <= 15:
            base_y = 4.5 * (1.35 - (self.shot_tick * 0.02))
            base_x = random.uniform(-0.4, 0.4)
        elif self.shot_tick <= 45:
            base_y = 2.8
            base_x = random.uniform(-1.2, 1.2)
        else:
            base_y = 2.4
            base_x = random.uniform(-1.8, 1.8)
            
        scale = 0.62 if is_ads else 1.0
        return base_x * strength * scale, base_y * strength * scale
        
    def reset(self):
        self.shot_tick = 0

# ==========================================
# 3. 硬體驅動動態防退化核心
# ==========================================
class HardwareManager:
    def __init__(self):
        self.use_kmbox = False
        self.kmNet = None
        self.init_hardware()

    def init_hardware(self):
        try:
            import kmNet
            kmNet.init(cfg.kmbox_ip, cfg.kmbox_port, cfg.kmbox_uuid)
            self.kmNet = kmNet
            self.use_kmbox = True
            print("✅ [兼容] Kmbox 網路硬體通道已建立。")
        except Exception:
            print("⚠️ [兼容] 未偵測到 Kmbox，啟用系統原生高相容滑鼠模擬。")
            self.use_kmbox = False

    def move(self, x, y):
        if self.use_kmbox and self.kmNet is not None:
            try:
                self.kmNet.move(int(x), int(y))
            except Exception:
                self.use_kmbox = False 
                ctypes.windll.user32.mouse_event(0x0001, int(x), int(y), 0, 0)
        else:
            ctypes.windll.user32.mouse_event(0x0001, int(x), int(y), 0, 0)

hardware = HardwareManager()

def is_key_down(vk_code):
    return (ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000) != 0

class TargetStabilizer:
    def __init__(self):
        self.prev_x, self.prev_y = 0, 0
        self.is_tracking = False
    def reset(self):
        self.is_tracking = False
    def stabilize(self, raw_x, raw_y, alpha, deadzone):
        dist = math.hypot(raw_x, raw_y)
        if dist < deadzone: return 0, 0
        if not self.is_tracking:
            self.prev_x, self.prev_y = raw_x, raw_y
            self.is_tracking = True
        else:
            self.prev_x = (self.prev_x * (1 - alpha)) + (raw_x * alpha)
            self.prev_y = (self.prev_y * (1 - alpha)) + (raw_y * alpha)
        damping = 0.5 if dist < 40 else 1.0
        return self.prev_x * damping, self.prev_y * damping

# ==========================================
# 4. Flask 輕量化網頁控制台 (整合熱鍵與硬體選擇)
# ==========================================
web_app = Flask(__name__)
HTML_TEMPLATE = """
<!DOCTYPE html><html><head><title>Delta AI 極致整合主控台</title><meta name="viewport" content="width=device-width, initial-scale=1">
<style>body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0c0c0e; color: #e4e4e7; padding: 15px; margin: 0; } .card { background: #18181b; padding: 20px; border-radius: 12px; max-width: 480px; margin: 20px auto; box-shadow: 0 4px 20px rgba(0,0,0,0.5); } h2 { text-align: center; color: #10b981; margin: 0 0 5px 0; font-size: 22px; } .subtitle { text-align: center; color: #71717a; margin-bottom: 20px; font-size: 13px; } .group { margin-bottom: 15px; background: #27272a; padding: 12px; border-radius: 8px; } label { display: block; font-weight: 600; margin-bottom: 6px; font-size: 14px; color:#d4d4d8; } input[type=range], select { width: 100%; accent-color: #10b981; } select { background: #18181b; color: #e4e4e7; border: 1px solid #3f3f46; padding: 8px; border-radius: 6px; font-size: 14px; outline: none; cursor: pointer; } .val { float: right; color: #10b981; font-family: monospace; } .btn-group { display:flex; justify-content:space-between; gap: 8px; margin-top: 4px; } button { flex: 1; padding: 10px; border: none; border-radius: 6px; font-weight: bold; cursor:pointer; font-size: 13px; transition: all 0.2s; } .act { background: #10b981; color: #09090b; box-shadow: 0 0 8px rgba(16,185,129,0.4); } .inact { background: #3f3f46; color: #a1a1aa; } .aim-card { border: 1px solid #10b98130; background: #10b98105; } .rage-card { border: 1px solid #ef444450; background: #ef444410; } .sniper-card { border: 1px solid #a855f750; background: #a855f70a; } .part-card { border: 1px solid #3b82f650; background: #3b82f60a; }</style></head><body>
<div class="card"><h2>Delta AI 終極整合版</h2><div class="subtitle">動態硬體變更 + 異步熱鍵映射機制</div>

<div class="group aim-card"><label style="color: #10b981;">🎯 AI 智慧吸附主開關</label><div class="btn-group"><button id="aim_on" class="inact" onclick="setParam('aimbot_enabled','1')">開啟吸附 (ON)</button><button id="aim_off" class="act" onclick="setParam('aimbot_enabled','0')">關閉吸附 (OFF)</button></div></div>
<div class="group aim-card"><label style="color: #34d399;">🔄 吸附模式切換</label><div class="btn-group"><button id="cont_on" class="inact" onclick="setParam('continuous_aim','1')">持續吸附 (無需按鍵)</button><button id="cont_off" class="act" onclick="setParam('continuous_aim','0')">熱鍵觸發 (點擊鎖定)</button></div></div>

<div class="group aim-card"><label style="color: #34d399;">⌨️ 觸發熱鍵設定</label>
<select id="hotkey_select" onchange="setParam('hotkey_name', this.value)">
    <option value="right_mouse" selected>滑鼠右鍵 (預設)</option>
    <option value="left_mouse">滑鼠左鍵</option>
    <option value="mouse_x1">滑鼠側鍵 X1</option>
    <option value="mouse_x2">滑鼠側鍵 X2</option>
    <option value="shift">鍵盤 Shift</option>
    <option value="alt">鍵盤 Alt</option>
</select></div>

<div class="group"><label style="color: #f59e0b;">⚡ 運算核心後端 (Device Backend)</label>
<select id="device_select" onchange="setParam('device_backend', this.value)">
    <option value="cuda">NVIDIA GPU (CUDA 加速)</option>
    <option value="cpu">中央處理器 (CPU 低耗模式)</option>
    <option value="npu">嵌入式 NPU (OpenVINO 強制編譯)</option>
</select></div>

<div class="group part-card"><label style="color: #60a5fa;">👤 優先鎖定部位</label>
<div class="btn-group">
    <button id="p0" class="inact" onclick="setParam('target_part','0')">頭部</button>
    <button id="p1" class="act" onclick="setParam('target_part','1')">胸部</button>
    <button id="p2" class="inact" onclick="setParam('target_part','2')">腹部</button>
    <button id="p3" class="inact" onclick="setParam('target_part','3')">手部</button>
    <button id="p4" class="inact" onclick="setParam('target_part','4')">腳部</button>
</div></div>

<div class="group sniper-card"><label style="color: #c084fc;">🔭 狙擊槍專用模式 (開鏡防震鎖頭)</label><div class="btn-group"><button id="snip_on" class="inact" onclick="setParam('sniper_mode','1')" style="color:#c084fc;">開啟狙擊模式</button><button id="snip_off" class="act" onclick="setParam('sniper_mode','0')">關閉狙擊模式</button></div></div>
<div class="group rage-card"><label style="color: #ef4444;">🔥 暴力模式 (無平滑死鎖)</label><div class="btn-group"><button id="rage_on" class="inact" onclick="setParam('rage_mode','1')" style="color:#f87171;">開啟暴力 (ON)</button><button id="rage_off" class="act" onclick="setParam('rage_mode','0')">關閉暴力 (OFF)</button></div></div>
<div class="group"><label>⭕ 鎖定範圍 FOV (像素): <span class="val" id="fov_v">100</span></label><input type="range" min="10" max="400" step="10" value="100" oninput="updateVol('fov', this.value, 'fov_v')"></div>
<div class="group"><label>靈敏度 (Smooth): <span class="val" id="smooth_v">0.35</span></label><input type="range" min="0.05" max="1.0" step="0.05" value="0.35" oninput="updateVol('smooth', this.value, 'smooth_v')"></div>
<div class="group"><label>⚡ 三角洲動態壓槍強度: <span class="val" id="rc_v">1.2</span></label><input type="range" min="0.0" max="3.0" step="0.1" value="1.2" oninput="updateVol('recoil_strength', this.value, 'rc_v')"></div>
<div class="group"><label>🎯 核心權重選擇 (.engine支援)</label><select id="model_select" onchange="setParam('model_file', this.value)"></select></div>
<div class="group"><label>🖥️ 實戰渲染視窗 (零耗能關閉)</label><div class="btn-group"><button id="w_off" class="act" onclick="setParam('show_window','0')">無視窗 (省電安全)</button><button id="w_on" class="inact" onclick="setParam('show_window','1')">開啓視窗</button></div></div>
<div class="group"><label>自動壓槍開關</label><div class="btn-group"><button id="recoil_on" class="inact" onclick="setParam('recoil_enabled','1')">開啓</button><button id="recoil_off" class="act" onclick="setParam('recoil_enabled','0')">關閉</button></div></div>
</div><script>
function setParam(p, v) { 
    fetch('/set?'+p+'='+encodeURIComponent(v)).then(() => {
        if(p==='aimbot_enabled'){ document.getElementById('aim_on').className = v==='1'?'act':'inact'; document.getElementById('aim_off').className = v==='0'?'act':'inact'; }
        else if(p==='continuous_aim'){ document.getElementById('cont_on').className = v==='1'?'act':'inact'; document.getElementById('cont_off').className = v==='0'?'act':'inact'; }
        else if(p==='sniper_mode'){ document.getElementById('snip_on').className = v==='1'?'act':'inact'; document.getElementById('snip_off').className = v==='0'?'act':'inact'; }
        else if(p==='rage_mode'){ document.getElementById('rage_on').className = v==='1'?'act':'inact'; document.getElementById('rage_off').className = v==='0'?'act':'inact'; }
        else if(p==='show_window'){ document.getElementById('w_on').className = v==='1'?'act':'inact'; document.getElementById('w_off').className = v==='0'?'act':'inact'; }
        else if(p==='recoil_enabled'){ document.getElementById('recoil_on').className = v==='1'?'act':'inact'; document.getElementById('recoil_off').className = v==='0'?'act':'inact'; }
        else if(p==='target_part'){
            for(let i=0; i<=4; i++){ document.getElementById('p'+i).className = (i==v) ? 'act' : 'inact'; }
        }
    });
}
function updateVol(p, v, id) { document.getElementById(id).innerText = v; fetch('/set?'+p+'='+v); }
fetch('/api/get_config').then(r => r.json()).then(data => {
    const select = document.getElementById('model_select');
    data.models.forEach(m => {
        let opt = document.createElement('option'); opt.value = m; opt.innerText = m;
        if (m === data.current_model) opt.selected = true; select.appendChild(opt);
    });
    document.getElementById('device_select').value = data.current_device;
});
</script></body></html>
"""
@web_app.route('/')
def index(): return render_template_string(HTML_TEMPLATE)

@web_app.route('/api/get_config')
def api_get_config():
    with cfg.lock: 
        return jsonify({
            "models": cfg.available_models, 
            "current_model": cfg.model_file,
            "current_device": cfg.device
        })

@web_app.route('/set')
def set_param():
    with cfg.lock:
        if 'aimbot_enabled' in request.args: cfg.aimbot_enabled = (request.args.get('aimbot_enabled') == '1')
        if 'continuous_aim' in request.args: cfg.continuous_aim = (request.args.get('continuous_aim') == '1')
        if 'sniper_mode' in request.args: cfg.sniper_mode = (request.args.get('sniper_mode') == '1')
        if 'rage_mode' in request.args: cfg.rage_mode = (request.args.get('rage_mode') == '1')
        if 'target_part' in request.args: cfg.target_part = int(request.args.get('target_part')) 
        if 'fov' in request.args: cfg.fov = int(request.args.get('fov'))
        if 'smooth' in request.args: cfg.smooth = float(request.args.get('smooth'))
        if 'recoil_strength' in request.args: cfg.recoil_strength = float(request.args.get('recoil_strength'))
        if 'recoil_enabled' in request.args: cfg.recoil_enabled = (request.args.get('recoil_enabled') == '1')
        if 'show_window' in request.args: cfg.show_window = (request.args.get('show_window') == '1')
        if 'model_file' in request.args:
            target_model = request.args.get('model_file')
            if target_model in cfg.available_models: cfg.model_file = target_model
        
        # 💡 [整合] 處理控制台發送的動態熱鍵更新
        if 'hotkey_name' in request.args:
            name = request.args.get('hotkey_name')
            if name in cfg.key_map: cfg.hotkey_code = cfg.key_map[name]
            
        # 💡 [新增] 處理控制台發送的硬體變更更新
        if 'device_backend' in request.args:
            dev = request.args.get('device_backend')
            if dev in ["cuda", "cpu", "npu"]: cfg.device = dev
            
    return jsonify({"status": "ok"})

# ==========================================
# 5. YOLO 高相容 O(N) 多硬體推論調度核心
# ==========================================
def get_optimized_model(model_path, device_target):
    """ 根據配置載入對應的模型與後端 """
    # 處理標準 CUDA / CPU
    if device_target == 'cuda' and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return YOLO(model_path, task='detect').to('cuda'), "cuda"
    return YOLO(model_path, task='detect').to('cpu'), "cpu"

def ai_loop():
    stabilizer = TargetStabilizer()
    recoil_sys = DeltaForceRecoilSystem() 
    
    current_device_setting = None
    current_model_setting = None
    model = None
    runtime_device = "cpu"
    window_created = False
    win_name = "Delta Pure Console"
    part_names = ["Head", "Chest", "Abdomen", "Arms", "Legs"]

    with mss.MSS() as sct:
        monitor_info = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        screen_w, screen_h = monitor_info["width"], monitor_info["height"]

        while True:
            loop_start = time.perf_counter()

            with cfg.lock:
                box = cfg.box_size; part = cfg.target_part; smooth = cfg.smooth
                noise = cfg.human_noise; rc_enabled = cfg.recoil_enabled; rc_str = cfg.recoil_strength
                target_device = cfg.device; show_win = cfg.show_window; max_fps = cfg.max_fps
                alpha_val = cfg.alpha; deadzone_val = cfg.deadzone; target_model = cfg.model_file
                aimbot_on = cfg.aimbot_enabled; fov = cfg.fov; rage_on = cfg.rage_mode
                continuous_on = cfg.continuous_aim  
                sniper_on = cfg.sniper_mode          
                active_hotkey = cfg.hotkey_code # 讀取動態熱鍵碼

            # 💡 [新增] 動態模型重載機制 (支援 NPU/OpenVINO 異步切換)
            if target_device != current_device_setting or target_model != current_model_setting:
                try:
                    if model is not None:
                        del model
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        gc.collect()
                        
                    if target_device == "npu":
                        # NPU 處理邏輯：檢查並自動編譯 OpenVINO 後端
                        model_base = os.path.splitext(target_model)[0]
                        openvino_dir = os.path.join(BASE_DIR, f"{model_base}_openvino_model")
                        if not os.path.exists(openvino_dir):
                            print(f"📦 首次對 NPU 初始化：正在將 {target_model} 編譯為 OpenVINO 後端...")
                            temp_model = YOLO(os.path.join(BASE_DIR, target_model))
                            temp_model.export(format="openvino")
                            del temp_model
                        model = YOLO(openvino_dir, task='detect')
                        runtime_device = "NPU"
                        print("✅ NPU (OpenVINO) 硬體加速核心載入成功！")
                    else:
                        model, runtime_device = get_optimized_model(os.path.join(BASE_DIR, target_model), target_device)
                        print(f"✅ 硬體後端已成功切換至: {runtime_device.upper()}")
                        
                    current_device_setting, current_model_setting = target_device, target_model
                except Exception as ex:
                    print(f"❌ 硬體後端切換出錯: {ex}，自動安全降級回 CPU 模式。")
                    cfg.device = "cpu"
                    continue

            # 讀取按鍵狀態
            left_down = is_key_down(0x01)
            hotkey_down = is_key_down(active_hotkey) # 💡 監聽動態熱鍵

            is_sniping_now = sniper_on and is_key_down(0x02) # 狙擊防震維持右鍵開鏡判定
            if is_sniping_now and not rage_on:
                fov = max(15, int(fov * 0.4))        
                smooth = smooth * 0.3               
                alpha_val = min(1.0, alpha_val * 1.5) 

            center_x, center_y = box // 2, box // 2
            top, left = (screen_h - box) // 2, (screen_w - box) // 2
            
            raw_bgra = np.array(sct.grab({"top": top, "left": left, "width": box, "height": box}), copy=False)
            frame = cv2.cvtColor(raw_bgra, cv2.COLOR_BGRA2BGR)
            
            try:
                # OpenVINO 不需要 half 參數，PyTorch CUDA 才需要
                is_half = (runtime_device == 'cuda')
                results = model.predict(frame, verbose=False, imgsz=box, half=is_half, device=runtime_device, max_det=5)[0]
            except Exception:
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                continue
            
            recoil_x_offset, recoil_y_offset = 0, 0
            if rc_enabled and left_down:
                recoil_x_offset, recoil_y_offset = recoil_sys.get_offset(rc_str, is_key_down(0x02))
            else:
                recoil_sys.reset()

            best_target_x, best_target_y = None, None
            min_dist_sq = fov ** 2
            
            boxes_data = results.boxes
            if boxes_data is not None and len(boxes_data) > 0:
                confs = boxes_data.conf.cpu().numpy()
                xyxys = boxes_data.xyxy.cpu().numpy()
                
                active_part = 0 if is_sniping_now else part
                
                for i in range(len(confs)):
                    if confs[i] > 0.45:
                        x1, y1, x2, y2 = xyxys[i]
                        w_box = x2 - x1
                        h_box = y2 - y1
                        
                        cx = int(x1 + w_box * 0.5)
                        if active_part == 0:    cy = int(y1 + h_box * 0.12)
                        elif active_part == 1:  cy = int(y1 + h_box * 0.28)
                        elif active_part == 2:  cy = int(y1 + h_box * 0.50)
                        elif active_part == 3:
                            cx = int(x1 + w_box * 0.75) 
                            cy = int(y1 + h_box * 0.35)
                        elif active_part == 4:  cy = int(y1 + h_box * 0.85)
                        else:                   cy = int(y1 + h_box * 0.50)
                        
                        dist_sq = (cx - center_x)**2 + (cy - center_y)**2
                        if dist_sq <= min_dist_sq:
                            min_dist_sq = dist_sq
                            best_target_x, best_target_y = cx, cy
                        
                        if show_win:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

            aim_x, aim_y = 0, 0
            if best_target_x is not None:
                if rage_on:
                    aim_x = best_target_x - center_x
                    aim_y = best_target_y - center_y
                else:
                    stable_x, stable_y = stabilizer.stabilize(best_target_x - center_x, best_target_y - center_y, alpha_val, deadzone_val)
                    aim_x = stable_x * smooth
                    aim_y = stable_y * smooth
                    if noise > 0 and math.hypot(stable_x, stable_y) > 5:
                        aim_x += random.uniform(-noise, noise)
                        aim_y += random.uniform(-noise, noise)
                        
                if show_win:
                    line_col = (0, 0, 255) if rage_on else (0, 165, 255)
                    cv2.circle(frame, (best_target_x, best_target_y), 5, (0, 0, 255), -1)
                    cv2.line(frame, (center_x, center_y), (best_target_x, best_target_y), line_col, 2 if rage_on else 1)
            else:
                stabilizer.reset()

            # 💡 [整合] 修正吸附觸發條件判定：若非持續吸附，則嚴格限制必須按下網頁設定的動態熱鍵
            if continuous_on:
                fire_aim = aimbot_on or rage_on
            else:
                fire_aim = (aimbot_on or rage_on) and hotkey_down

            final_x = (aim_x if fire_aim else 0) + recoil_x_offset
            final_y = (aim_y if fire_aim else 0) + recoil_y_offset
            
            if final_x != 0 or final_y != 0: 
                hardware.move(final_x, final_y)

            if show_win:
                if not window_created:
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(win_name, 350, 350)
                    cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
                    window_created = True
                
                fov_color = (255, 0, 255) if is_sniping_now else ((0, 0, 255) if rage_on else (16, 185, 129))
                cv2.circle(frame, (center_x, center_y), fov, fov_color, 2 if is_sniping_now else 1)
                
                status_text = "SNIPER" if is_sniping_now else ("RAGE" if rage_on else ("CONT" if continuous_on else "HOTKEY"))
                current_part_name = "Head (Sniper Override)" if is_sniping_now else part_names[part]
                cv2.putText(frame, f"Mode: {status_text} | Target: {current_part_name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, fov_color, 1)
                cv2.imshow(win_name, frame)
            else:
                if window_created:
                    cv2.destroyWindow(win_name)
                    window_created = False
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            if max_fps > 0:
                elapsed = time.perf_counter() - loop_start
                target_period = 1.0 / max_fps
                if elapsed < target_period: time.sleep(target_period - elapsed)

if __name__ == '__main__':
    # 啟動網頁伺服器
    threading.Thread(target=lambda: web_app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False), daemon=True).start()
    print("🌐 控制台已整合完畢。請訪問網址：http://127.0.0.1:5000")
    ai_loop()