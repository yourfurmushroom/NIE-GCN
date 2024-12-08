import random

def split_data_by_edges(input_file, train_file, test_file, train_ratio=0.9):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    user_edges = {i: [] for i in range(47)}

    for line in lines:
        parts = line.split()
        user_id = int(parts[0])
        edges = parts[1:]
        user_edges[user_id] = edges

    train_lines = []
    test_lines = []

    for user_id in range(47):
        edges = user_edges[user_id]
        if len(edges) > 1:
            random.shuffle(edges)
            split_index = int(len(edges) * train_ratio)
            train_edges = edges[:split_index]
            test_edges = edges[split_index:]
            train_lines.append(f"{user_id} {' '.join(train_edges)}\n")
            test_lines.append(f"{user_id} {' '.join(test_edges)}\n")
        elif len(edges) == 1:
            train_lines.append(f"{user_id} {' '.join(edges)}\n")
            test_lines.append(f"{user_id}\n")
        else:
            train_lines.append(f"{user_id}\n")
            test_lines.append(f"{user_id}\n")

    with open(train_file, 'w') as file:
        file.writelines(train_lines)

    with open(test_file, 'w') as file:
        file.writelines(test_lines)

if __name__ == "__main__":
    input_file = 'user_to_restaurant_0.9.txt'
    train_file = 'Data/food/train.txt'
    test_file = 'Data/food/test.txt'
    split_data_by_edges(input_file, train_file, test_file, train_ratio=0.5)