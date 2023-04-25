# n=max(0.1,(1*0.9999**(21600/32*33)))
# print(n)

def normalize(data):
    max_val = max(data)
    min_val = min(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data


data = [9738, 12430, 11380,
        8760, 10248, 9260,
        5762, 7063, 7260,
        3760, 5036, 6140,
        3084, 3927, 4486]

result = normalize(data)
print(result)
