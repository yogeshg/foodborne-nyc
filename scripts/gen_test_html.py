#!/usr/bin/env python

gradient_1 = [0.0 + x/20.0 for x in range(10)]
gradient_255 = [255 * x/10.0 for x in range(10)]

with open('b.html', 'w') as f:
  f.write('<h1>text-color</h1>\n')
  f.write('<h2>blue > red</h2>\n')
  for r in gradient_1:
    for b in gradient_1:
      f.write(
        '<span \nstyle="background-color: rgba(0, 0, 255, {b});">'
        '<span \nstyle="background-color: rgba(255, 0, 0, {r});">'
        '#'
        '</span>'
        '</span>'.format(r=r,b=b))
    f.write('</br>')

  for r in gradient_1:
    b = 0
    f.write(
      '<span \nstyle="background-color: rgba(0, 0, 255, {b});">'
      '<span \nstyle="background-color: rgba(255, 0, 0, {r});">'
      '#'
      '</span>'
      '</span>'.format(r=r,b=b))
  f.write('</br>')
  for b in gradient_1:
    r = 0
    f.write(
      '<span \nstyle="background-color: rgba(0, 0, 255, {b});">'
      '<span \nstyle="background-color: rgba(255, 0, 0, {r});">'
      '#'
      '</span>'
      '</span>'.format(r=r,b=b))
  f.write('</br>')
  for b in gradient_1:
    r = b
    f.write(
      '<span \nstyle="background-color: rgba(0, 0, 255, {b});">'
      '<span \nstyle="background-color: rgba(255, 0, 0, {r});">'
      '#'
      '</span>'
      '</span>'.format(r=r,b=b))
  f.write('</br>')

  f.write('<h2>red > blue</h2>\n')
  for r in gradient_1:
    for b in gradient_1:
      f.write(
        '<span \nstyle="background-color: rgba(255, 0, 0, {r});">'
        '<span \nstyle="background-color: rgba(0, 0, 255, {b});">'
        '#'
        '</span>'
        '</span>'.format(r=r,b=b))
    f.write('</br>')
  for r in gradient_1:
    b = 0
    f.write(
      '<span \nstyle="background-color: rgba(255, 0, 0, {r});">'
      '<span \nstyle="background-color: rgba(0, 0, 255, {b});">'
      '#'
      '</span>'
      '</span>'.format(r=r,b=b))
  f.write('</br>')
  for b in gradient_1:
    r = 0
    f.write(
      '<span \nstyle="background-color: rgba(255, 0, 0, {r});">'
      '<span \nstyle="background-color: rgba(0, 0, 255, {b});">'
      '#'
      '</span>'
      '</span>'.format(r=r,b=b))
  f.write('</br>')
  for b in gradient_1:
    r = b
    f.write(
      '<span \nstyle="background-color: rgba(255, 0, 0, {r});">'
      '<span \nstyle="background-color: rgba(0, 0, 255, {b});">'
      '#'
      '</span>'
      '</span>'.format(r=r,b=b))
  f.write('</br>')

  f.write('<h1>background-color</h1>\n')
  for r in gradient_255:
    for b in gradient_255:
      f.write('<span \nstyle="color: rgb(255, 255, 255); background-color:rgb({r}, 0, {b});">#</span>'.format(r=r,b=b))
    f.write('</br>')

  f.write('<h1>text-color</h1>\n')
  for r in gradient_255:
    for b in gradient_255:
      f.write('<span \nstyle="color: rgb({r}, 0, {b});">#</span>'.format(r=r,b=b))
    f.write('</br>')

