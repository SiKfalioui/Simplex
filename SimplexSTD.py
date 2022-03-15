import numpy as np
import time

# start_time = time.time() si on veut comptabiliser le temps de saisie ce qui contre representatif a mon avis
precision = 3
np.set_printoptions(precision=precision, suppress=True)  # fixer la precision
contraintes = []
i = 1
print(
    "\n******************************************************************************************************************"
    "\nNB: Ce programme resout uniquement des problémes lineaires sous forme Ax<=B, avec B>=0"
    "\n"
    "******************************************************************************************************************\n")
try:
    min_max = int(
        input('Taper \"0\" pour minimiser, \"1\" pour maximiser: \n---------------------------------------------\n'))
    if min_max in (0, 1):
        pass
    else:
        raise Exception('La valeur doit etre 0 ou 1')  # pour le cas ou min_max pas 0 ou 1
except ValueError:
    raise Exception('La valeur doit etre 0 ou 1')  # pour le cas ou int(str)

dict_mm = {0: 'minimisation', 1: 'maximisation'}
fnobjectif = np.array(  # pour pouvoir accepter les fractions  :)
    [float(item) if '/' not in item else float(float(item.split('/')[0]) / float(item.split('/')[1])) for item in
     input(
         "Entrer les coefficients de la function objectif : \n-------------------------------------------------\n", ).split()])
nbrdecision = len(fnobjectif)
print(
    "Entrer les contraintes:\n--------------------------\n  \"\"\"Par exemple pour 4x+2z<=5 entrer 4 2 5\"\"\" \n   "
    "-Enter \"*\" pour finir la saisie des contraintes: ")
saisie = True
while saisie:
    try:
        contraintes.append(np.array(
            [float(item) if '/' not in item else float(float(item.split('/')[0]) / float(item.split('/')[1])) for item
             in input(f"\ncontrainte {i}:\n------------\n", ).split()]))  # pour pouvoir accepter les fractions :)
        if len(contraintes[i - 1]) != nbrdecision + 1:
            raise Exception('mauvaise dimension des contraintes')
        i += 1
    except ValueError:
        saisie = False
        print('\nFin de la saisie des contraintes.\n')
        pass
start_time = time.time()
nbrcontraintes = i - 1
contraintes = np.array(contraintes)
b = np.array(contraintes[:, -1])  # second membre
if False in (b >= 0):
    raise ValueError('Valeurs de b inférieures a zero')
Identite_b = np.column_stack((np.identity(nbrcontraintes, dtype=float), b))  # Identite et second membre
compfnob = np.append([fnobjectif], np.zeros(nbrcontraintes + 1, dtype=float))  # fonction objectif suivie de zeros
a_resoudre = np.column_stack((np.append(np.column_stack((contraintes[:, :nbrdecision], Identite_b)),
                                        np.array([compfnob]), axis=0), np.zeros([nbrcontraintes + 1, 1], dtype=float)))
a_resoudre = np.array(a_resoudre)
entrant_sortant = {}
solutions = np.zeros(nbrdecision + 1)  # pour les valeurs des vars et z
solutions[-1] = a_resoudre[nbrcontraintes, -2]  # valeur de z dans le tableau


def strsol(sol, nbrdecis):
    variables_str = '(' + ', '.join(['X' + str(n + 1) for n in range(nbrdecis)]) + ')  =  '  # variables
    solutions_str = '(' + ', '.join([str(round(j, precision)) for j in sol[:-1]]) + ')'  # solutions
    return variables_str + solutions_str


print(f'Résolution du probléme de {dict_mm[min_max]} pour le PL: '
      f'\n------------------------------------------------------\n')
print(a_resoudre,
      "\n\nSolution de base: {x} avec Z={z}\n----------------------------------------------------------------------\n\n".format(
          x=strsol(solutions, nbrdecision),
          z=round(solutions[-1], precision)))
if min_max:
    bool_list = a_resoudre[-1, :-2] <= 0  # pour la maximisation
else:
    bool_list = a_resoudre[-1, :-2] >= 0  # pour la minimisation
while False in bool_list:
    filtre = np.zeros(nbrdecision, dtype=float)
    Bland_valeurs = []
    if min_max:
        MaxMinCol = a_resoudre[-1, :-2].argmax()  # cherche le maximum des cofficients de la fn objecti]
    else:
        MaxMinCol = a_resoudre[-1, :-2].argmin()  # cherche le minimum des cofficients de la fn objecti
    #minligne = np.where(a_resoudre[:-1, -1] > 0, a_resoudre[:-1, -1], np.inf).argmin()
    c = (a_resoudre[:-1, -2] / a_resoudre[:-1, MaxMinCol] == np.amin(np.where(a_resoudre[:-1, -2] / a_resoudre[:-1, MaxMinCol]>0,
                     a_resoudre[:-1, -2] / a_resoudre[:-1, MaxMinCol],np.inf))).sum()
    # calcul si il y'a des elements qui se repete dans b/cp
    # ce qui cause le recours a la regle de bland
    # c est une liste contenant le nombre de repetition de chaque element dans b/cp
    while (c > 1):
        print("utilisation de la regle de Bland: changement de pivot...\n")
        Bland_valeurs.append(MaxMinCol)
        if len(Bland_valeurs) == nbrdecision:
            raise Exception('Pas de solutions calculable possible')
        for p in range(nbrdecision):
            if min_max:
                filtre[p] = a_resoudre[-1, p] if (
                        p not in Bland_valeurs) else np.NINF  # verifie les indexes qui sont dans la liste de bland
            else:
                filtre[p] = a_resoudre[
                    -1, p] if p not in Bland_valeurs else np.inf  # verifie les indexes qui sont dans la liste de
                # bland
        if min_max:
            MaxMinCol = filtre.argmax()  # cherche un coef qui n'est pas dans la liste de bland
        else:
            MaxMinCol = filtre.argmin()
        c = (a_resoudre[:-1, -2] / a_resoudre[:-1, MaxMinCol] == np.amin(
            np.where(a_resoudre[:-1, -2] / a_resoudre[:-1, MaxMinCol] > 0,
                     a_resoudre[:-1, -2] / a_resoudre[:-1, MaxMinCol], np.inf))).sum()
    a_resoudre[:-1, -1] = a_resoudre[:-1, -2] / a_resoudre[:-1, MaxMinCol]  # b/cp
    minli = np.where(a_resoudre[:-1, -1] > 0, a_resoudre[:-1, -1], np.inf).argmin()  # cherche le minmum de b/cp
    entrant_sortant[MaxMinCol] = minli
    a_resoudre[minli] /= a_resoudre[minli, MaxMinCol]  # ligne pivot
    if minli != 0:
        a_resoudre[:minli] -= np.transpose(
            np.outer(a_resoudre[minli], a_resoudre[:minli, MaxMinCol]))  # elimination en dessus de la ligne pivot
    a_resoudre[minli + 1:] -= np.transpose(
        np.outer(a_resoudre[minli], a_resoudre[minli + 1:, MaxMinCol]))  # elimination en dessous de la ligne pivot
    solutions[MaxMinCol] = a_resoudre[minli, -2]
    solutions[-1] = -a_resoudre[nbrcontraintes, -2]
    if min_max:
        bool_list = a_resoudre[-1, :-2] <= 0  # pour la maximisation
    else:
        bool_list = a_resoudre[-1, :-2] >= 0  # pour la minimisation
    if False not in bool_list:
        for k, v in entrant_sortant.items():
            solutions[k] = a_resoudre[v, -2]  # pour le cas ou les contraintes sont toutes respectées
        print(a_resoudre,
              "\n\nSolution finale: {x} avec Z={z:.3f}"
              "\n--------------------------------------------------------------\n".format(
                  x=strsol(solutions, nbrdecision),
                  z=round(solutions[-1], precision)))
    else:
        print(a_resoudre,
              "\n\nSolution actuelle: {x} avec Z={z:.3f}"
              "\n--------------------------------------------------------------\n".format(
                  x=strsol(solutions, nbrdecision),
                  z=round(solutions[-1], precision)))

print("---temps d'execution:  %s s ---" % (time.time() - start_time))