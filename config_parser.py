import configparser


def parse_file(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    settings = {}
    bool_list = ['true','false']
    for key,value in (config['General'].items()):
        if value.lower() == bool_list[0]:
            settings[key] = True
        elif value.lower() == bool_list[1]:
            settings[key] = False
        elif (value.isdigit()):
            settings[key] = int(value)
        else:
            settings[key] = value

    for key,value in (config['model'].items()):
        settings[key] = float(value)

    for key,value in (config['train'].items()):
        if (value.isdigit()):
            settings[key] = int(value)
        else:
            settings[key] = float(value)

    for key,value in (config['data'].items()):
        settings[key] = value

    for key,value in (config['output'].items()):
        settings[key] = value
    for k,v in settings.items():
        print(k,':',v)
    return settings

if __name__ == '__main__':
    config_file = parse_file('config.ini')
    print(config_file)