import subprocess

def run_script(script_name):
    """运行一个Python脚本"""
    try:
        # 使用 subprocess.run 来运行脚本
        _ = subprocess.run(['python', script_name], check=True)
        print(f"{script_name} 执行成功")
    except subprocess.CalledProcessError as e:
        print(f"{script_name} 执行失败: {e}")
        exit(1)  # 如果某个脚本失败，则停止整个流程

def main():
    script1 = 'training/train_PCA.py'
    script2 = 'training/train_autoencoder_per.py'
    script3 = 'training/train_autoencoder_por.py'
    script4 = 'training/train_surrogateModel.py'
    script5 = 'scripts/dataAssimilation.py'

    scripts = [script1, script2, script3, script4, script5]

    for script in scripts:
        run_script(script)

if __name__ == "__main__":
    main()
