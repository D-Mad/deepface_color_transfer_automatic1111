import launch

if not launch.is_installed("numpy"):
    launch.run_pip("install numpy", "requirements for Color Transfer")

if not launch.is_installed("opencv-python"):
    launch.run_pip("install opencv-python", "requirements for Color Transfer")
    
if not launch.is_installed("deepface"):
    launch.run_pip("install deepface", "requirements for Color Transfer")    
if not launch.is_installed("tensorfow"):
    launch.run_pip("install tensorflow==2.15.0", "requirements for Color Transfer")   
if not launch.is_installed("tf-keras"):
    launch.run_pip("install tf-keras==2.15.0", "requirements for Color Transfer") 