

with open('b.html', 'a') as f:
  for r in gradient:
    for b in gradient:
      f.write('<span \nstyle="color: rgb(255, 255, 255); background-color:rgb({r}, 0, {b});">#</span>'.format(r=r,b=b))
    f.write('</br>')
