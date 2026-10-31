import os
root = "data_resampled/200hz"
print(os.getcwd())
print(os.path.exists("data_resampled"), os.path.exists(root))
if os.path.exists("data_resampled"):
    print(os.listdir("data_resampled"))
if os.path.exists(root):
    print(os.listdir(root)[:10])
    for dp, dn, fn in os.walk(root):
        print(dp, dn[:3], fn[:5])
        break