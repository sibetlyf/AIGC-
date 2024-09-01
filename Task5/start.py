import subprocess


def start_app():
    subprocess.Popen(['python', 'stream/app.py'])

def start_backend():
    subprocess.Popen(['python', 'stream/backend.py'])

if __name__ == '__main__':
    start_app()
    start_backend()
    print("Both app.py and backend.py are running.")