import matplotlib.pyplot as plt

# 1. Gender distribution
labels1 = ['Female', 'Male']
sizes1 = [3, 16]

plt.figure(figsize=(5,5))
plt.pie(sizes1, labels=labels1, autopct='%1.1f%%', startangle=90)
plt.title('Gender Distribution')
plt.axis('equal')
plt.show()

# 2. Age distribution
labels2 = ['18–20', '21–23', '24–28']
sizes2 = [2, 13, 4]

plt.figure(figsize=(5,5))
plt.pie(sizes2, labels=labels2, autopct='%1.1f%%', startangle=90)
plt.title('Age Distribution')
plt.axis('equal')
plt.show()

# 3. Eye Trackin Experience
labels3 = ['Low', 'Medium', 'High']
sizes3 = [14, 4, 1]

plt.figure(figsize=(5,5))
plt.pie(sizes3, labels=labels3, autopct='%1.1f%%', startangle=90)
plt.title('Eye Tracking Experience')
plt.axis('equal')
plt.show()

# 4. Brought own laptop device
labels4 = ['Yes', 'No']
sizes4 = [6, 13]

plt.figure(figsize=(5,5))
plt.pie(sizes4, labels=labels4, autopct='%1.1f%%', startangle=90)
plt.title('Own Device')
plt.axis('equal')
plt.show()

# 5. Vision Correction
labels5 = ['None', 'Contact lenses']
sizes5 = [14, 5]

plt.figure(figsize=(5,5))
plt.pie(sizes5, labels=labels5, autopct='%1.1f%%', startangle=90)
plt.title('Vision Correction')
plt.axis('equal')
plt.show()
