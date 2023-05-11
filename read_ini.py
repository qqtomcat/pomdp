import configparser
def read_ini(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    configuration={}
    for section in config.sections():
        for key in config[section]:
            pair = {key : config[section][key]} 
            configuration.update(pair)
                        
    return configuration

