s = 'https://search.jd.com/Search?keyword=%E6%A4%85%E5%AD%90&enc=utf-8&qrst=1&rt=1&stop=1&vt=2&wq=%E6%A4%85%E5%AD%90&page=3&s=54&click=0'
q = 'https://search.jd.com/Search?keyword=%E6%A4%85%E5%AD%90&enc=utf-8&qrst=1&rt=1&stop=1&vt=2&wq=%E6%A4%85%E5%AD%90&page=1&s=58&click=0'

res = str("椅子".encode('utf8'))
res = res.replace('x', '').replace('b\'', '').replace('\'', '').upper().replace('\\', '%')
print(res)