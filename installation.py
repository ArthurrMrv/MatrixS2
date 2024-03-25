# To install all the required dependencies, the Matrix module and create a python venv please run this script in the terminal:

import os

def install():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    os.system(f"cd {current_dir}")
    
    if not(".venv" in os.listdir(current_dir)):
        os.system("python3 -m venv .venv")
    os.system("source .venv/bin/activate")
    os.system("pip3 install Flask")
    os.system("pip3 install Flask-WTF")
    os.system("pip3 install numpy")
    os.system("pip3 install matplotlib")
    os.system("pip3 install scipy")

if __name__ == "__main__":
    ans = ''
    for _ in range(3):
        ans = input("Do you want to install the dependencies and create a python venv? (Y/n): ").lower()
        if ans == 'y':
            install()
            break
        elif ans == 'n':
            break
        else:
            pass
    raise Exception("Initialiszation aborted")