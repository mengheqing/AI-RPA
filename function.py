def get_legal_path(path):
    path = path.replace('\\', '/')
    if path[-1] != '/':
        return path + '/'
    else:
        return path