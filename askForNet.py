import requests
r=requests.get("https://juejin.cn/book/6857911863016390663/section/6870393485346734091")
print(r.encoding)
print(r.text)
print(r.headers)
v=r.cookies["cookie_name"]
print(v)