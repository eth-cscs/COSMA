#!/usr/bin/env python3
import fileinput

f = open('result.csv','r')
temp = f.read()
f.close()

f = open('result.csv', 'w')
f.write("m,n,k,p,algorithm,time,case\n")

f.write(temp)
f.close()

with fileinput.FileInput('result.csv', inplace=True) as file:
    for line in file:
        print(line.replace('old_carma', 'CARMA [22] ').replace('carma', 'COSMM (this work) ').replace('scalapack', 'ScaLAPACK [52] ').replace('cyclops', 'CTF [48] '), end='')

with fileinput.FileInput('result.csv', inplace=True) as file:
    for line in file:
       print(line.replace('CARMA [22] ', 'CARMA [21] ').replace('COSMM (this work)', 'COSMA (this work) ').replace('ScaLAPACK [52]', 'ScaLAPACK [51] ').replace('CTF [48]', 'CTF [47] '), end='')
