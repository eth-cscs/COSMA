import os

# mapa (dictionary) koja opisuje koji string se menja u koji
transformations = {
        "#include <costa/grid2grid/": "#include <costa/grid2grid/grid2grid/",
        }

# ova funkcija primenjuje transformacije na sve fajlove (rekurzivno) koji su:
# - unutar rootdir
# - imaju ekstenziju .php
# - koji nisu iz foldera adminer/editor (# posto to nije nas kod)
def replace(tranformations):
    #################################################
    # OVE PROMENLJIVE SE MOGU PRILAGODITI PO POTREBI
    #################################################
    # podrazumevamo da smo u DOCUMENT_ROOT
    rootdir = './'

    #################################################
    # PRIMENA TRANSFORMACIJA NA SVAKI FAJL
    #################################################
    # obidji sve fajlove u rootdir (= DOCUMENT_ROOT)
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            # originalni fajl
            file_name = os.path.join(subdir, file)
            # novi fajl (za uradjenim transformacijama = .php.tmp)
            new_file_name = file_name + ".tmp"

            # naziv fajla koji se trenutno obradjuje
            print ("Processing file: " + file_name)

            # ucitaj originalni fajl za citanje
            f = open(file_name, 'r')
            lines = f.readlines()
            f.close()

            # ucitaj novi fajl za pisanje
            f = open(new_file_name, 'w')
            for line in lines:
                # prodji kroz sve transformacije
                for pattern in transformations:
                    # primeni transformaciju
                    line = line.replace(pattern, transformations[pattern])
                # kada su sve transformacije primenjene
                # upisi ovu liniju u novi fajl
                f.write(line)
            f.close();

            # ukloni originalni fajl
            os.remove(file_name);
            # preimenuj novi fajl u naziv originalnog fajla (.php.tmp -> .php)
            os.rename(new_file_name, file_name);

# primeni sve transformacije
# main funkcija
if __name__ == '__main__':
    replace(transformations)

