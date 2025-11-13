import matplotlib.pyplot as plt

first_color = "#FFA239"
second_color = "#4E61D3"
third_color = "#98A1BC"

# Function to apply white, bold text for autopct
def make_autopct(pcts):
    def my_autopct(pct):
        total = sum(pcts)
        val = int(round(pct*total/100.0))
        return f"{pct:.1f}%\n({val})"
    return my_autopct

# 1. Gender distribution
labels1 = ['Female', 'Male']
sizes1 = [3, 16]
colors1 = [second_color,first_color]  # Example colors

plt.figure(figsize=(5,5))
wedges, texts, autotexts = plt.pie(
    sizes1, labels=labels1, autopct=make_autopct(sizes1),
    startangle=90, colors=colors1, textprops={'color':'white','weight':'bold'}
)
plt.title('Gender Distribution')
plt.axis('equal')
plt.figtext(0.5, 0.05, f"{labels1[0]} ({sizes1[0]})    |    {labels1[1]} ({sizes1[1]})",
            ha="center", fontsize=12, weight='bold', color="gray")
plt.show()


# 2. Age distribution
labels2 = ['19-21', '22–24', '25–28']
sizes2 = [4, 12, 3]
colors2 = [third_color, first_color, second_color]

plt.figure(figsize=(5,5))
wedges, texts, autotexts = plt.pie(
    sizes2, labels=labels2, autopct=make_autopct(sizes2),
    startangle=30, colors=colors2, textprops={'color':'white','weight':'bold'}
)
plt.title('Age Distribution')
plt.axis('equal')
plt.figtext(0.5, 0.05, f"{labels2[0]} ({sizes2[0]})    |    {labels2[1]} ({sizes2[1]})    |    {labels2[2]} ({sizes2[2]})",
            ha="center", fontsize=12, weight='bold', color="gray")
plt.show()


# 3. Eye Tracking Experience
labels3 = ['Low', 'Medium', 'High']
sizes3 = [15, 4, 0]
colors3 = [first_color, second_color, third_color]

plt.figure(figsize=(5,5))
wedges, texts, autotexts = plt.pie(
    sizes3, labels=labels3, autopct=make_autopct(sizes3),
    startangle=60, colors=colors3, textprops={'color':'white','weight':'bold'}
)
plt.title('Eye Tracking Experience')
plt.axis('equal')
plt.figtext(0.5, 0.05, f"{labels3[0]} ({sizes3[0]})    |    {labels3[1]} ({sizes3[1]})    |    {labels3[2]} ({sizes3[2]})",
            ha="center", fontsize=12, weight='bold', color="gray")
plt.show()


# 4. Own Device
labels4 = ['Yes', 'No']
sizes4 = [6, 13]
colors4 = [second_color, first_color]

plt.figure(figsize=(5,5))
wedges, texts, autotexts = plt.pie(
    sizes4, labels=labels4, autopct=make_autopct(sizes4),
    startangle=60, colors=colors4, textprops={'color':'white','weight':'bold'}
)
plt.title('Own Device')
plt.axis('equal')
plt.figtext(0.5, 0.05, f"{labels4[0]} ({sizes4[0]})    |    {labels4[1]} ({sizes4[1]})",
            ha="center", fontsize=12, weight='bold', color="gray")
plt.show()


# 5. Vision Correction
labels5 = ['None', 'Contact lenses']
sizes5 = [14, 5]
colors5 = [first_color, second_color]

plt.figure(figsize=(5,5))
wedges, texts, autotexts = plt.pie(
    sizes5, labels=labels5, autopct=make_autopct(sizes5),
    startangle=-10, colors=colors5, textprops={'color':'white','weight':'bold'}
)
plt.title('Vision Correction')
plt.axis('equal')
plt.figtext(0.5, 0.05, f"{labels5[0]} ({sizes5[0]})    |    {labels5[1]} ({sizes5[1]})",
            ha="center", fontsize=12, weight='bold', color="gray")
plt.show()
