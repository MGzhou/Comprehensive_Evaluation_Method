import os

root_path = os.path.abspath(__file__)

a = os.path.dirname(root_path)

b = os.path.dirname(a)

print(root_path)
print(a)
print(b)