import subprocess, shutil

def save_gif(input_folder, save_folder = None, save_name = 'out.gif', ww=256, framerate=12):
    if save_folder is None:
        save_folder = input_folder
        
    command = ['ffmpeg', '-y', '-r', '%d' % framerate, '-i', input_folder + 'frame_%03d.png',
               '-filter_complex', '[0:v]scale=w=%d:h=-2,split [a][b];[a] palettegen=stats_mode=diff [p];[b][p] paletteuse=new=1' % ww,
               input_folder + 'aa_out.gif']

    p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = p.communicate()
    rc = p.returncode
    
#     print(err)
    
    shutil.move(input_folder + 'aa_out.gif', save_folder + save_name)
    
    return p


def save_gif_windows(input_folder, save_folder = None, save_name = 'out.gif', ww=256, framerate=12):
    if save_folder is None:
        save_folder = input_folder
        
    command = ['C:\\ffmpeg\\bin\\ffmpeg.exe', '-y', '-r', '%d' % framerate, '-i', input_folder + 'frame_%03d.png',
               '-filter_complex', '[0:v]scale=w=%d:h=-2,split [a][b];[a] palettegen=stats_mode=diff [p];[b][p] paletteuse=new=1' % ww,
               input_folder + 'aa_out.gif']

    p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = p.communicate()
    rc = p.returncode
    
#     print(err)
    
    shutil.move(input_folder + 'aa_out.gif', save_folder + save_name)
    
    return p