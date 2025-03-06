import subprocess
import threading

def play_audio(path):
    subprocess.Popen(["ffplay", "-nodisp", "-autoexit", path])
