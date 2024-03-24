import os

def launch_browser(host = "127.0.0.1", port = 5000):
    
    os.system(f"open http://{host}:{port}")
    os.system(f"python3 web/app.py {host} {port}")
    

if __name__ == "__main__":
    launch_browser()