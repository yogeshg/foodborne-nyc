import sh

def save_code():
    try:
        o = sh.git.diff('--cached', '--name-only', '--exit-code')
    except Exception as e:
        sh.git.commit('-m', 'auto commit staged files')
    try:
        o = sh.git.diff('--name-only', '--exit-code')
    except Exception as e:
        sh.git.commit('-am', 'auto commit tracked files')
    o = sh.git('rev-parse', 'HEAD').strip()
    return o

