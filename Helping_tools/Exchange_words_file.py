search_text = 'position_translater'

replace_text = 'sorted_names_nodes'

with open('Generating_Problems.py', 'r') as file:
    data = file.read()
    data = data.replace(search_text, replace_text)

with open('Generating_Problems.py', 'w') as file:
    file.write(data)


with open('Calculating_Expectation_Values.py', 'r') as file:
    data = file.read()
    data = data.replace(search_text, replace_text)

with open('QIRO_MIS.py', 'w') as file:
    file.write(data)


with open('QIRO_MAX_2SAT.py', 'r') as file:
    data = file.read()
    data = data.replace(search_text, replace_text)

with open('GQIRO_MAX_2SAT.py', 'w') as file:
    file.write(data)

print("Text replaced")
