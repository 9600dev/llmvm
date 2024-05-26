
import toml

def read_requirements(file_path):
    requirements = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            requirements.append(line)
    return requirements

def parse_requirement(requirement):
    if '==' in requirement:
        package, version = requirement.split('==', 1)
    elif '>=' in requirement:
        package, version = requirement.split('>=', 1)
    elif '<=' in requirement:
        package, version = requirement.split('<=', 1)
    elif '>' in requirement:
        package, version = requirement.split('>', 1)
    elif '<' in requirement:
        package, version = requirement.split('<', 1)
    elif '~=' in requirement:
        package, version = requirement.split('~=', 1)
    else:
        package = requirement
        version = "*"
    return package, version

def update_pyproject_toml(requirements, pyproject_path):
    pyproject = toml.load(pyproject_path)
    
    if 'tool' not in pyproject:
        pyproject['tool'] = {}
    if 'poetry' not in pyproject['tool']:
        pyproject['tool']['poetry'] = {}
    if 'dependencies' not in pyproject['tool']['poetry']:
        pyproject['tool']['poetry']['dependencies'] = {}
    
    for req in requirements:
        package, version = parse_requirement(req)
        pyproject['tool']['poetry']['dependencies'][package] = version

    with open(pyproject_path, 'w') as file:
        toml.dump(pyproject, file)

requirements = read_requirements('../requirements.txt')
update_pyproject_toml(requirements, '../pyproject.toml')

