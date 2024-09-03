import os

def reset_internals(): 
    with open("vars/internal_ids.txt", "w") as file:
        file.write("0\n0")

def get_internal_author_id():
    with open("vars/internal_ids.txt", "r") as file:
        author_id = file.readline().strip()
    
    return author_id

def get_internal_paper_id():
    with open("vars/internal_ids.txt", "r") as file:
        file.readline()
        paper_id = file.readline().strip()
    
    return paper_id

def increment_internal_author_id():
    with open("vars/internal_ids.txt", 'r') as file:
        lines = file.readlines()

    first_value = int(lines[0].strip())
    first_value += 1
    lines[0] = str(first_value) + "\n"

    with open("vars/internal_ids.txt", 'w') as file:
        file.writelines(lines)


def increment_internal_paper_id():
    with open("vars/internal_ids.txt", 'r') as file:
        lines = file.readlines()

    second_value = int(lines[1].strip())
    second_value += 1
    lines[1] = str(second_value) + "\n"

    with open("vars/internal_ids.txt", 'w') as file:
        file.writelines(lines)


if __name__ == "__main__":
    reset_internals()