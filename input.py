# Input data (queries)
X = [
    "select * from users",
    "or 1=1 --",
    "normal input",
    "drop table users",
    "hello world"
]

# Output labels
# 0 = Safe , 1 = Injection
y = [0, 1, 0, 1, 0]

