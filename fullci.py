import numpy as np
import opt_einsum as oe
import itertools as it
from pyscf import gto, scf, mcscf
import time

mol = gto.M(
    verbose = 0,
    atom = """
Li                 0.00000000    0.00000000    0.00000000
H                  0.00000000    0.00000000    1.60000000
    """,
    basis = 'sto-3g',
    charge = 0,
    spin = 0
)

t1 = time.time()

Enuc = gto.energy_nuc(mol)
# print("Enuc =", Enuc)
nao = gto.nao_nr(mol, cart=None)
print("nao =", nao)
# print(type(ao))
norb = nao
nelec = gto.tot_electrons(mol)
print("nelec =", nelec)
occ = nelec//2

rhf_pyscf = scf.RHF(mol)
Erhf_pyscf = rhf_pyscf.kernel()
# print("Erhf_pyscf =", Erhf_pyscf)
mo_coeff = rhf_pyscf.mo_coeff
# print(mo_coeff)
mo_energy = rhf_pyscf.mo_energy
# print(mo_energy)

fullci_pyscf = mcscf.CASSCF(rhf_pyscf, norb, nelec).run()
Efullci_pyscf = fullci_pyscf.e_tot
print("Efullci_pyscf =", Efullci_pyscf)

# integrals
ovlp = mol.intor('int1e_ovlp')
# print(ovlp.shape)
kin = mol.intor('int1e_kin')
# print(kin)
nuc = mol.intor('int1e_nuc')
# print(nuc)
int1e = kin + nuc
# print(int1e)
int2e = mol.intor('int2e')
#print(int2e)

# transformation matrix
T = np.zeros((nao, nao))
val, vec = np.linalg.eigh(ovlp)
# print(val)
# print(vec)
t = val**(-1/2)
for i in range(nao):
    T[i][i] = t[i] 
# print(T)
Xsym = np.dot(np.dot(vec, T), np.conjugate(vec.T))
# print(Xsym)

Xsym_inv = np.linalg.inv(Xsym)
# print(Xsym_inv)

ovlp_ortho = oe.contract("it,ju,ij->tu", Xsym, Xsym, ovlp)
int1e_ortho = oe.contract("it,ju,ij->tu", Xsym, Xsym, int1e)
int2e_ortho = oe.contract("it,ju,kv,lw,ijkl->tuvw", Xsym, Xsym, Xsym, Xsym, int2e)

# molecular orbital coefficient
mo_coeff_ortho = np.dot(Xsym_inv, mo_coeff)
# print(mo_coeff_ortho)

t2 = time.time()

int1e_mo = oe.contract("tp,uq,tu->pq", mo_coeff_ortho, mo_coeff_ortho, int1e_ortho)
int2e_mo = oe.contract("tp,uq,vr,ws,tuvw->pqrs", mo_coeff_ortho, mo_coeff_ortho, mo_coeff_ortho, mo_coeff_ortho, int2e_ortho)

al = be = nelec//2
alpha = [1]*al + [0]*(norb-al)
# print(alpha)
# print(type(norb))
# print(type(alpha))
# print(type(int(norb)))
alpha = list(it.permutations(alpha,int(norb)))
# print(alpha)
con_alpha = sorted(set(alpha), key=alpha.index)
# print("alpha configuration")
# print(con_alpha)
num_con_alpha = len(con_alpha)
# print(num_alpha)

con_beta = con_alpha
num_con_beta = num_con_alpha
# print("num_beta", num_beta)

con_ele = []
for a in con_alpha:
    for b in con_beta:
        each_con_ele = a + b
        con_ele.append(each_con_ele)
# print("electron configuration")
# print(con_ele)
num_con_ele = len(con_ele)
# print(num_con_ele)

dec = np.zeros((num_con_ele, num_con_ele))
for i in range(num_con_ele):
    for j in range(num_con_ele):
        de = 0.0
        for z in range(2*norb):
            de += con_ele[i][z] * con_ele[j][z]
        dec[i][j] = de
        # print(i, j, dec[i][j])
# print("dec")
# print(dec)
# print(dec.shape)

""" Construction of the Hamiltonian """
bra = []
ket = []
bra_ex = []
ket_ex = []
unex = []

H = np.zeros((num_con_ele, num_con_ele))
for i in range(num_con_ele):
    for j in range(num_con_ele):
    
        # n = i*num_con_ele + j

        " ground state "
        if dec[i][j]==nelec:

            bra_gs = []
            ket_gs = []
            for z in range(2*norb):
                if con_ele[i][z]==1:
                    bra_gs.append(z)
            bra.append(bra_gs)
            for z in range(2*norb):
                if con_ele[j][z]==1:
                    ket_gs.append(z)
            ket.append(ket_gs)

            bra_ex.append([])
            ket_ex.append([])

            unex_gs = []
            for z in range(2*norb):
                if con_ele[i][z]==con_ele[j][z]==1:
                    unex_gs.append(z)
            unex.append(unex_gs)

            H0aa = H0bb = 0.0
            for t in bra_gs:
                if t>=norb:
                    # bb
                    H0bb += int1e_mo[t-norb][t-norb]
                else:
                    # aa
                    H0aa += int1e_mo[t][t]

            two00 = list(it.combinations(bra_gs,2))
            # print(two00)

            W0aa = W0bb = W0ab = W0ba = 0.0
            for conb in two00:
                # aaaa
                if conb[0]<norb and conb[1]<norb:
                    W0aa += int2e_mo[conb[0]][conb[0]][conb[1]][conb[1]] - int2e_mo[conb[0]][conb[1]][conb[1]][conb[0]]
                # bbbb
                elif conb[0]>=norb and conb[1]>=norb:
                    W0bb += int2e_mo[conb[0]-norb][conb[0]-norb][conb[1]-norb][conb[1]-norb] - int2e_mo[conb[0]-norb][conb[1]-norb][conb[1]-norb][conb[0]-norb]
                # aabb
                elif conb[0]<norb and conb[1]>=norb:
                    W0ab += int2e_mo[conb[0]][conb[0]][conb[1]-norb][conb[1]-norb]
                # bbaa
                elif conb[0]>=norb and conb[1]<norb:
                    W0ba += int2e_mo[conb[0]-norb][conb[0]-norb][conb[1]][conb[1]]

            H[i][j] = H0aa + H0bb + W0aa + W0bb + W0ab + W0ba
            # print(H[i][j])

        " one electron excited state"
        if dec[i][j]==nelec-1:

            bra_one = []
            ket_one = []
            for z in range(2*norb):
                if con_ele[i][z]==1:
                    bra_one.append(z)
            bra.append(bra_one)
            for z in range(2*norb):
                if con_ele[j][z]==1:
                    ket_one.append(z)
            ket.append(ket_one)

            bra_ex_one = []
            ket_ex_one = []
            for z in range(2*norb):
                if con_ele[i][z]!=con_ele[j][z]:
                    if con_ele[i][z]==1:
                        bra_ex_one.append(z)
                    if con_ele[j][z]==1:
                        ket_ex_one.append(z)
            bra_ex.append(bra_ex_one)
            ket_ex.append(ket_ex_one)

            unex_one = []
            for z in range(2*norb):
                if con_ele[i][z]==con_ele[j][z]==1:
                    unex_one.append(z)
            unex.append(unex_one)

            # bb
            if bra_ex_one[0]>=norb and ket_ex_one[0]>=norb:
                W1_bb = 0.0
                for t in bra_one:
                    W1bb = 0.0
                    # bbbb
                    if t>=norb:
                        W1bb = int2e_mo[bra_ex_one[0]-norb][ket_ex_one[0]-norb][t-norb][t-norb] - int2e_mo[bra_ex_one[0]-norb][t-norb][t-norb][ket_ex_one[0]-norb]
                    # bbaa
                    elif t<norb:
                        W1bb = int2e_mo[bra_ex_one[0]-norb][ket_ex_one[0]-norb][t][t]
                    # print("each")
                    # print(W1bb)
                    W1_bb += W1bb
                # print("bb")
                # print(W1bb)

                H[i][j] = int1e_mo[bra_ex_one[0]-norb][ket_ex_one[0]-norb] + W1_bb
                # print(int1e_mo[bra_ex_one[0]-norb][ket_ex_one[0]-norb])
                # print(W1bb)

            # aa
            elif bra_ex_one[0]<norb and ket_ex_one[0]<norb:
                W1_aa = 0.0
                for t in bra_one:
                    W1aa = 0.0
                    # aaaa
                    if t<norb:
                        W1aa = int2e_mo[bra_ex_one[0]][ket_ex_one[0]][t][t] - int2e_mo[bra_ex_one[0]][t][t][ket_ex_one[0]]
                    # aabb
                    elif t>=norb:
                        W1aa = int2e_mo[bra_ex_one[0]][ket_ex_one[0]][t-norb][t-norb]
                    W1_aa += W1aa
                # print("aa")
                # print(W1aa)

                H[i][j] = int1e_mo[bra_ex_one[0]][ket_ex_one[0]] + W1_aa
                # print(int1e_mo[bra_ex_one[0]][ket_ex_one[0]])
                # print(W1aa)

            else:
                H[i][j] = 0.0

            for p in range(nelec):
                for q in range(nelec):
                    if bra_one[p]==bra_ex_one[0] and ket_one[q]==ket_ex_one[0]:
                        if (p-q)%2==0:
                            H[i][j] = + H[i][j]
                        else:
                            H[i][j] = - H[i][j]

        " two electron excited state "
        if dec[i][j]==nelec-2:

            bra_two = []
            ket_two = []
            for z in range(2*norb):
                if con_ele[i][z]==1:
                    bra_two.append(z)
            bra.append(bra_two)
            for z in range(2*norb):
                if con_ele[j][z]==1:
                    ket_two.append(z)
            ket.append(ket_two)

            bra_ex_two = []
            ket_ex_two = []
            for z in range(2*norb):
                if con_ele[i][z]!=con_ele[j][z]:
                    if con_ele[i][z]==1:
                        bra_ex_two.append(z)
                    if con_ele[j][z]==1:
                        ket_ex_two.append(z)
            bra_ex.append(bra_ex_two)
            ket_ex.append(ket_ex_two)	

            unex_two = []
            for z in range(2*norb):
                if con_ele[i][z]==con_ele[j][z]==1:
                    unex_two.append(z)
            unex.append(unex_two)
                            
            # bbbb
            if bra_ex_two[0]>=norb and ket_ex_two[0]>=norb and bra_ex_two[1]>=norb and ket_ex_two[1]>=norb:
                H[i][j] = int2e_mo[bra_ex_two[0]-norb][ket_ex_two[0]-norb][bra_ex_two[1]-norb][ket_ex_two[1]-norb] - int2e_mo[bra_ex_two[0]-norb][ket_ex_two[1]-norb][bra_ex_two[1]-norb][ket_ex_two[0]-norb]
            # aabb
            elif bra_ex_two[0]<norb and ket_ex_two[0]<norb and bra_ex_two[1]>=norb and ket_ex_two[1]>=norb:
                H[i][j] = int2e_mo[bra_ex_two[0]][ket_ex_two[0]][bra_ex_two[1]-norb][ket_ex_two[1]-norb]
            # aaaa
            elif bra_ex_two[0]<norb and ket_ex_two[0]<norb and bra_ex_two[1]<norb and ket_ex_two[1]<norb:
                H[i][j] = int2e_mo[bra_ex_two[0]][ket_ex_two[0]][bra_ex_two[1]][ket_ex_two[1]] - int2e_mo[bra_ex_two[0]][ket_ex_two[1]][bra_ex_two[1]][ket_ex_two[0]]

            for p in range(nelec):
                for q in range(nelec):
                    for r in range(nelec):
                        for s in range(nelec):
                            if bra_two[p]==bra_ex_two[0] and ket_two[q]==ket_ex_two[0] and bra_two[r]==bra_ex_two[1] and ket_two[s]==ket_ex_two[1]:
                                if ((p-q)%2==0 and (r-s)%2==0) or ((p-q)%2!=0 and (r-s)%2!=0):
                                    H[i][j] = + H[i][j]
                                else:
                                    H[i][j] = - H[i][j]

        " more three electron excited state "
        if dec[i][j]<=nelec-3:
            H[i][j] = 0.0

            bra_more = []
            ket_more = []
            for z in range(2*norb):
                if con_ele[i][z]==1:
                    bra_more.append(z)
            bra.append(bra_more)
            for z in range(2*norb):
                if con_ele[j][z]==1:
                    ket_more.append(z)
            ket.append(ket_more)

            bra_ex_more = []
            ket_ex_more = []
            for z in range(2*norb):
                if con_ele[i][z]!=con_ele[j][z]:
                    if con_ele[i][z]==1:
                        bra_ex_more.append(z)
                    if con_ele[j][z]==1:
                        ket_ex_more.append(z)
            bra_ex.append(bra_ex_more)
            ket_ex.append(ket_ex_more)

            unex_more = []
            for z in range(2*norb):
                if con_ele[i][z]==con_ele[j][z]==1:
                    unex_more.append(z)
            unex.append(unex_more)
                    
        # print("bra_i", n, bra_i[n])
        # print("ket_j", n, ket_j[n])

        # print("bra_i_ex", n, bra_i_ex[n])
        # print("ket_j_ex", n, ket_j_ex[n])

# print("bra")
# print(bra)
# print("ket")
# print(ket)

# print("bra_ex")
# print(bra_ex)
# print("ket_ex")
# print(ket_ex)

# print("unex")
# print(unex)
    
        # print(i, j, H[i][j])

# print("H")
# print(H)

val, vec = np.linalg.eigh(H)
# print(val)
# print(vec)
id = np.argsort(val)
val = val[id]
vec = vec[:,id]
# print(val)
# print(vec)
Efullci_elec = val[0]
# print("Efullci_elec =", Efullci_elec)
Efullci = Enuc + Efullci_elec
print("Efullci =", Efullci)

t3 = time.time()
eig_time = t3 - t2
print(f"eig_time: {eig_time}")

ci = vec[:,0].reshape(num_con_alpha, num_con_beta)

Dci = np.zeros((norb, norb))
for p in range(norb):
    for q in range(norb):
        Dci[p][q] = 0.0

        if p==q:
            for i in range(num_con_alpha):
                for j in range(num_con_beta):
                    for k in range(num_con_alpha):
                        for l in range(num_con_beta):
                            if i==k and j==l and i==j:
                                if con_alpha[k][p]==1 and con_beta[l][p]==1:
                                    Dci[p][q] += 2 * ci[i][j] * ci[k][l]
                            if i==k and j==l and i!=j:
                                if con_alpha[k][p]==1 and con_beta[l][p]==1:
                                    Dci[p][q] += 2 * ci[i][j] * ci[k][l]
                                elif con_alpha[k][p]==1:
                                    Dci[p][q] += ci[i][j] * ci[k][l]
                                elif con_beta[l][p]==1:
                                    Dci[p][q] += ci[i][j] * ci[k][l]
                            # print(p, q, Dci[p][q], i, j, k, l)

        elif p!=q:
            for i in range(num_con_alpha):
                for j in range(num_con_beta):
                    for k in range(num_con_alpha):
                        for l in range(num_con_beta):

                            " one electron excited state "
                            #b
                            if i==k and j!=l:
                                j_beta = []
                                for spin in range(norb):
                                    if con_beta[j][spin]==1:
                                        j_beta.append(spin) 
                                # print("j", j_beta)
                                l_beta = []
                                for spin in range(norb):
                                    if con_beta[l][spin]==1:
                                        l_beta.append(spin)

                                j_con_beta = []
                                for spin in con_beta[j]:
                                    j_con_beta.append(spin)
                                l_con_beta = []
                                for spin in con_beta[l]:
                                    l_con_beta.append(spin)
                                # print("before", l_con_beta)        
                                l_con_beta[q] = 0
                                l_con_beta[p] = 1
                                # print("after", l_con_beta)

                                if j_con_beta==l_con_beta:
                                    for a in range(be):
                                        for b in range(be):
                                            if j_beta[a]==p and l_beta[b]==q:
                                                if (a-b)%2==0:
                                                    Dci[p][q] += ci[i][j] * ci[k][l]
                                                    # print("positive1")
                                                elif (a-b)%2!=0:
                                                    Dci[p][q] += - ci[i][j] * ci[k][l]
                                                    # print("negative1")
                            #a
                            if i!=k and j==l:
                                i_alpha = []
                                for spin in range(norb):
                                    if con_alpha[i][spin]==1:
                                        i_alpha.append(spin) 
                                # print("i", i_alpha)
                                k_alpha = []
                                for spin in range(norb):
                                    if con_alpha[k][spin]==1:
                                        k_alpha.append(spin)

                                i_con_alpha = []
                                for spin in con_alpha[i]:
                                    i_con_alpha.append(spin)
                                k_con_alpha = []
                                for spin in con_alpha[k]:
                                    k_con_alpha.append(spin)
                                # print("before", k_con_alpha)        
                                k_con_alpha[q] = 0
                                k_con_alpha[p] = 1
                                # print("after", k_con_alpha)

                                if i_con_alpha==k_con_alpha:
                                    for a in range(al):
                                        for b in range(al):
                                            if i_alpha[a]==p and k_alpha[b]==q:
                                                if (a-b)%2==0:
                                                    Dci[p][q] += ci[i][j] * ci[k][l]
                                                    # print("positive2")
                                                elif (a-b)%2!=0:
                                                    Dci[p][q] += - ci[i][j] * ci[k][l]
                                                    # print("negative2")

                            # print(p, q, Dci[p][q], i, j, k, l)

        # print(p, q, Dci[p][q])

# print("Dci")
# print(Dci)

nci = 0.0
for i in range(norb):
    nci += Dci[i][i]
# print("nci =", nci)

Pci = np.zeros((norb, norb, norb, norb))
for p in range(norb):
    for q in range(norb):
        for r in range(norb):
            for s in range(norb):
                
                Pci[p][q][r][s] = 0.0

                for a in range(num_con_alpha):
                    for b in range(num_con_beta):
                        for c in range(num_con_alpha):
                            for d in range(num_con_beta):

                                i = a*num_con_alpha + b
                                j = c*num_con_alpha + d

                                # print(dec)	

                                n = i*num_con_ele + j

                                " ground state "
                                if dec[i][j]==nelec:
                                    
                                    # print(i)
                                    # print(j)			
                                    # print(bra[n])
                                    # print(ket[n])

                                    con_unex = list(it.combinations(unex[n], 2))
                                    # print(con_unex)
                                    len_con_unex = len(con_unex)
                                    # print(len_con_unex)
                                    
                                    #+1221,+2112,-1212,-2121
                                    #1:a,2:a
                                    for x in range(len_con_unex):
                                        if p==q==con_unex[x][0] and r==s==con_unex[x][1]:
                                            Pci[p][q][r][s] += ci[a][b] * ci[c][d]
                                        if p==q==con_unex[x][1] and r==s==con_unex[x][0]:
                                            Pci[p][q][r][s] += ci[a][b] * ci[c][d]
                                        if p==s==con_unex[x][0] and r==q==con_unex[x][1]:
                                            Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                            # print("c", Pci[p][q][r][s])
                                        if p==s==con_unex[x][1] and r==q==con_unex[x][0]:
                                            Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                            # print("d", Pci[p][q][r][s])

                                    #1:a,2:b										
                                    for x in range(len_con_unex):
                                        if p==q==con_unex[x][0] and r==s==con_unex[x][1]-norb:
                                            Pci[p][q][r][s] += ci[a][b] * ci[c][d]
                                            # print("a", Pci[p][q][r][s])
                                        if p==q==con_unex[x][1]-norb and r==s==con_unex[x][0]:
                                            Pci[p][q][r][s] += ci[a][b] * ci[c][d]
                                            # print("b", Pci[p][q][r][s])

                                    #1:b,2:b
                                    for x in range(len_con_unex):
                                        if p==q==con_unex[x][0]-norb and r==s==con_unex[x][1]-norb:
                                            Pci[p][q][r][s] += ci[a][b] * ci[c][d]
                                        if p==q==con_unex[x][1]-norb and r==s==con_unex[x][0]-norb:
                                            Pci[p][q][r][s] += ci[a][b] * ci[c][d]
                                        if p==s==con_unex[x][0]-norb and r==q==con_unex[x][1]-norb:
                                            Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                        if p==s==con_unex[x][1]-norb and r==q==con_unex[x][0]-norb:
                                            Pci[p][q][r][s] += - ci[a][b] * ci[c][d]

                                    # print("ground state",p,q,r,s,Pci[p][q][r][s])
                                    
                                " one electron excited state "
                                if dec[i][j]==nelec-1:

                                    # print(i)
                                    # print(j)
                                    # print(n)
                                    # print(dec[i][j])			
                                    # print(bra[n])
                                    # print(ket[n])
                                    # print(bra_ex[n])
                                    # print(ket_ex[n])
                                    # print(unex[n])

                                    len_unex = len(unex[n])

                                    for x in range(len_unex):
                                        # ex:aa, unex:aa
                                        if p==bra_ex[n][0] and q==ket_ex[n][0] and r==s==unex[n][x]:
                                            for br in range(nelec):
                                                for ke in range(nelec):
                                                    if bra[n][br]==bra_ex[n][0] and ket[n][ke]==ket_ex[n][0]:
                                                        if (br-ke)%2==0:
                                                            Pci[p][q][r][s] += ci[a][b] * ci[c][d]
                                                        if (br-ke)%2!=0:
                                                            Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                        if r==bra_ex[n][0] and s==ket_ex[n][0] and p==q==unex[n][x]:
                                            for br in range(nelec):
                                                for ke in range(nelec):
                                                    if bra[n][br]==bra_ex[n][0] and ket[n][ke]==ket_ex[n][0]:
                                                        if (br-ke)%2==0:
                                                            Pci[p][q][r][s] += ci[a][b] * ci[c][d]
                                                        if (br-ke)%2!=0:
                                                            Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                        if p==bra_ex[n][0] and s==ket_ex[n][0] and r==q==unex[n][x]:
                                            for br in range(nelec):
                                                for ke in range(nelec):
                                                    if bra[n][br]==bra_ex[n][0] and ket[n][ke]==ket_ex[n][0]:
                                                        if (br-ke)%2==0:
                                                            Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                                        if (br-ke)%2!=0:
                                                            Pci[p][q][r][s] += ci[a][b] * ci[c][d]
                                        if r==bra_ex[n][0] and q==ket_ex[n][0] and p==s==unex[n][x]:
                                            for br in range(nelec):
                                                for ke in range(nelec):
                                                    if bra[n][br]==bra_ex[n][0] and ket[n][ke]==ket_ex[n][0]:
                                                        if (br-ke)%2==0:
                                                            Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                                        if (br-ke)%2!=0:
                                                            Pci[p][q][r][s] += ci[a][b] * ci[c][d]

                                        # ex:aa, unex:bb
                                        if p==bra_ex[n][0] and q==ket_ex[n][0] and r==s==unex[n][x]-norb:
                                            for br in range(nelec):
                                                for ke in range(nelec):
                                                    if bra[n][br]==bra_ex[n][0] and ket[n][ke]==ket_ex[n][0]:
                                                        if (br-ke)%2==0:
                                                            Pci[p][q][r][s] += ci[a][b] * ci[c][d]
                                                        if (br-ke)%2!=0:
                                                            Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                            # print("1ab", Pci[p][q][r][s], a,b,c,d)
                                        if r==bra_ex[n][0] and s==ket_ex[n][0] and p==q==unex[n][x]-norb:
                                            for br in range(nelec):
                                                for ke in range(nelec):
                                                    if bra[n][br]==bra_ex[n][0] and ket[n][ke]==ket_ex[n][0]:
                                                        if (br-ke)%2==0:
                                                            Pci[p][q][r][s] += ci[a][b] * ci[c][d]
                                                        if (br-ke)%2!=0:
                                                            Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                            # print("2ab", Pci[p][q][r][s], a,b,c,d)
                                        
                                        # ex:bb, unex:aa
                                        if p==bra_ex[n][0]-norb and q==ket_ex[n][0]-norb and r==s==unex[n][x]:
                                            for br in range(nelec):
                                                for ke in range(nelec):
                                                    if bra[n][br]==bra_ex[n][0] and ket[n][ke]==ket_ex[n][0]:
                                                        if (br-ke)%2==0:
                                                            Pci[p][q][r][s] += ci[a][b] * ci[c][d]
                                                        if (br-ke)%2!=0:
                                                            Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                            # print("1ba", Pci[p][q][r][s], a,b,c,d)
                                        if r==bra_ex[n][0]-norb and s==ket_ex[n][0]-norb and p==q==unex[n][x]:
                                            for br in range(nelec):
                                                for ke in range(nelec):
                                                    if bra[n][br]==bra_ex[n][0] and ket[n][ke]==ket_ex[n][0]:
                                                        if (br-ke)%2==0:
                                                            Pci[p][q][r][s] += ci[a][b] * ci[c][d]
                                                        if (br-ke)%2!=0:
                                                            Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                            # print("2ba", Pci[p][q][r][s], a,b,c,d)

                                        # ex:bb, unex:bb
                                        if p==bra_ex[n][0]-norb and q==ket_ex[n][0]-norb and r==s==unex[n][x]-norb:
                                            for br in range(nelec):
                                                for ke in range(nelec):
                                                    if bra[n][br]==bra_ex[n][0] and ket[n][ke]==ket_ex[n][0]:
                                                        if (br-ke)%2==0:
                                                            Pci[p][q][r][s] += ci[a][b] * ci[c][d]
                                                        if (br-ke)%2!=0:
                                                            Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                        if r==bra_ex[n][0]-norb and s==ket_ex[n][0]-norb and p==q==unex[n][x]-norb:
                                            for br in range(nelec):
                                                for ke in range(nelec):
                                                    if bra[n][br]==bra_ex[n][0] and ket[n][ke]==ket_ex[n][0]:
                                                        if (br-ke)%2==0:
                                                            Pci[p][q][r][s] += ci[a][b] * ci[c][d]
                                                        if (br-ke)%2!=0:
                                                            Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                        if p==bra_ex[n][0]-norb and s==ket_ex[n][0]-norb and r==q==unex[n][x]-norb:
                                            for br in range(nelec):
                                                for ke in range(nelec):
                                                    if bra[n][br]==bra_ex[n][0] and ket[n][ke]==ket_ex[n][0]:
                                                        if (br-ke)%2==0:
                                                            Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                                        if (br-ke)%2!=0:
                                                            Pci[p][q][r][s] += ci[a][b] * ci[c][d]
                                        if r==bra_ex[n][0]-norb and q==ket_ex[n][0]-norb and p==s==unex[n][x]-norb:
                                            for br in range(nelec):
                                                for ke in range(nelec):
                                                    if bra[n][br]==bra_ex[n][0] and ket[n][ke]==ket_ex[n][0]:
                                                        if (br-ke)%2==0:
                                                            Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                                        if (br-ke)%2!=0:
                                                            Pci[p][q][r][s] += ci[a][b] * ci[c][d]

                                    # print("one",p,q,r,s,Pci[p][q][r][s])
                                                                
                                " two electron excited state "                    
                                if dec[i][j]==nelec-2:

                                    # print(i)
                                    # print(j)			
                                    # print(bra[n])
                                    # print(ket[n])
                                    # print(bra_ex[n])
                                    # print(ket_ex[n])
                                    # print(unex[n])

                                    # ex1:aa, ex2:aa
                                    if p==bra_ex[n][0] and q==ket_ex[n][0] and r==bra_ex[n][1] and s==ket_ex[n][1]:
                                        for b0 in range(nelec):
                                            for k0 in range(nelec):
                                                for b1 in range(nelec):
                                                    for k1 in range(nelec):
                                                        if bra[n][b0]==bra_ex[n][0] and ket[n][k0]==ket_ex[n][0] and bra[n][b1]==bra_ex[n][1] and ket[n][k1]==ket_ex[n][1]:
                                                            if ((b0-k0)%2==0 and (b1-k1)%2==0) or ((b0-k0)%2!=0 and (b1-k1)%2!=0):
                                                                Pci[p][q][r][s] += ci[a][b] * ci[c][d]
                                                            else:
                                                                Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                    if r==bra_ex[n][0] and s==ket_ex[n][0] and p==bra_ex[n][1] and q==ket_ex[n][1]:
                                        for b0 in range(nelec):
                                            for k0 in range(nelec):
                                                for b1 in range(nelec):
                                                    for k1 in range(nelec):
                                                        if bra[n][b0]==bra_ex[n][0] and ket[n][k0]==ket_ex[n][0] and bra[n][b1]==bra_ex[n][1] and ket[n][k1]==ket_ex[n][1]:
                                                            if ((b0-k0)%2==0 and (b1-k1)%2==0) or ((b0-k0)%2!=0 and (b1-k1)%2!=0):
                                                                Pci[p][q][r][s] += ci[a][b] * ci[c][d]
                                                            else:
                                                                Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                    if p==bra_ex[n][0] and s==ket_ex[n][0] and r==bra_ex[n][1] and q==ket_ex[n][1]:
                                        for b0 in range(nelec):
                                            for k0 in range(nelec):
                                                for b1 in range(nelec):
                                                    for k1 in range(nelec):
                                                        if bra[n][b0]==bra_ex[n][0] and ket[n][k0]==ket_ex[n][0] and bra[n][b1]==bra_ex[n][1] and ket[n][k1]==ket_ex[n][1]:
                                                            if ((b0-k0)%2==0 and (b1-k1)%2==0) or ((b0-k0)%2!=0 and (b1-k1)%2!=0):
                                                                Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                                            else:
                                                                Pci[p][q][r][s] += + ci[a][b] * ci[c][d]
                                    if r==bra_ex[n][0] and q==ket_ex[n][0] and p==bra_ex[n][1] and s==ket_ex[n][1]:
                                        for b0 in range(nelec):
                                            for k0 in range(nelec):
                                                for b1 in range(nelec):
                                                    for k1 in range(nelec):
                                                        if bra[n][b0]==bra_ex[n][0] and ket[n][k0]==ket_ex[n][0] and bra[n][b1]==bra_ex[n][1] and ket[n][k1]==ket_ex[n][1]:
                                                            if ((b0-k0)%2==0 and (b1-k1)%2==0) or ((b0-k0)%2!=0 and (b1-k1)%2!=0):
                                                                Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                                            else:
                                                                Pci[p][q][r][s] += + ci[a][b] * ci[c][d]

                                    # ex1:aa, ex2:bb
                                    if p==bra_ex[n][0] and q==ket_ex[n][0] and r==bra_ex[n][1]-norb and s==ket_ex[n][1]-norb:
                                        for b0 in range(nelec):
                                            for k0 in range(nelec):
                                                for b1 in range(nelec):
                                                    for k1 in range(nelec):
                                                        if bra[n][b0]==bra_ex[n][0] and ket[n][k0]==ket_ex[n][0] and bra[n][b1]==bra_ex[n][1] and ket[n][k1]==ket_ex[n][1]:
                                                            if ((b0-k0)%2==0 and (b1-k1)%2==0) or ((b0-k0)%2!=0 and (b1-k1)%2!=0):
                                                                Pci[p][q][r][s] += ci[a][b] * ci[c][d]
                                                            else:
                                                                Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                    if r==bra_ex[n][0] and s==ket_ex[n][0] and p==bra_ex[n][1]-norb and q==ket_ex[n][1]-norb:
                                        for b0 in range(nelec):
                                            for k0 in range(nelec):
                                                for b1 in range(nelec):
                                                    for k1 in range(nelec):
                                                        if bra[n][b0]==bra_ex[n][0] and ket[n][k0]==ket_ex[n][0] and bra[n][b1]==bra_ex[n][1] and ket[n][k1]==ket_ex[n][1]:
                                                            if ((b0-k0)%2==0 and (b1-k1)%2==0) or ((b0-k0)%2!=0 and (b1-k1)%2!=0):
                                                                Pci[p][q][r][s] += ci[a][b] * ci[c][d]
                                                            else:
                                                                Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                    
                                    # ex1:bb, ex2:bb
                                    if p==bra_ex[n][0]-norb and q==ket_ex[n][0]-norb and r==bra_ex[n][1]-norb and s==ket_ex[n][1]-norb:
                                        for b0 in range(nelec):
                                            for k0 in range(nelec):
                                                for b1 in range(nelec):
                                                    for k1 in range(nelec):
                                                        if bra[n][b0]==bra_ex[n][0] and ket[n][k0]==ket_ex[n][0] and bra[n][b1]==bra_ex[n][1] and ket[n][k1]==ket_ex[n][1]:
                                                            if ((b0-k0)%2==0 and (b1-k1)%2==0) or ((b0-k0)%2!=0 and (b1-k1)%2!=0):
                                                                Pci[p][q][r][s] += ci[a][b] * ci[c][d]
                                                            else:
                                                                Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                    if r==bra_ex[n][0]-norb and s==ket_ex[n][0]-norb and p==bra_ex[n][1]-norb and q==ket_ex[n][1]-norb:
                                        for b0 in range(nelec):
                                            for k0 in range(nelec):
                                                for b1 in range(nelec):
                                                    for k1 in range(nelec):
                                                        if bra[n][b0]==bra_ex[n][0] and ket[n][k0]==ket_ex[n][0] and bra[n][b1]==bra_ex[n][1] and ket[n][k1]==ket_ex[n][1]:
                                                            if ((b0-k0)%2==0 and (b1-k1)%2==0) or ((b0-k0)%2!=0 and (b1-k1)%2!=0):
                                                                Pci[p][q][r][s] += ci[a][b] * ci[c][d]
                                                            else:
                                                                Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                    if p==bra_ex[n][0]-norb and s==ket_ex[n][0]-norb and r==bra_ex[n][1]-norb and q==ket_ex[n][1]-norb:
                                        for b0 in range(nelec):
                                            for k0 in range(nelec):
                                                for b1 in range(nelec):
                                                    for k1 in range(nelec):
                                                        if bra[n][b0]==bra_ex[n][0] and ket[n][k0]==ket_ex[n][0] and bra[n][b1]==bra_ex[n][1] and ket[n][k1]==ket_ex[n][1]:
                                                            if ((b0-k0)%2==0 and (b1-k1)%2==0) or ((b0-k0)%2!=0 and (b1-k1)%2!=0):
                                                                Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                                            else:
                                                                Pci[p][q][r][s] += + ci[a][b] * ci[c][d]
                                    if r==bra_ex[n][0]-norb and q==ket_ex[n][0]-norb and p==bra_ex[n][1]-norb and s==ket_ex[n][1]-norb:
                                        for b0 in range(nelec):
                                            for k0 in range(nelec):
                                                for b1 in range(nelec):
                                                    for k1 in range(nelec):
                                                        if bra[n][b0]==bra_ex[n][0] and ket[n][k0]==ket_ex[n][0] and bra[n][b1]==bra_ex[n][1] and ket[n][k1]==ket_ex[n][1]:
                                                            if ((b0-k0)%2==0 and (b1-k1)%2==0) or ((b0-k0)%2!=0 and (b1-k1)%2!=0):
                                                                Pci[p][q][r][s] += - ci[a][b] * ci[c][d]
                                                            else:
                                                                Pci[p][q][r][s] += + ci[a][b] * ci[c][d]
                
                                    # print("two",p,q,r,s,Pci[p][q][r][s])
                
                # print(p,q,r,s," ",Pci[p][q][r][s])

# print("Pci")
# print(Pci)

t4 = time.time()
dm_time = t4 - t3
print(f"dm_time: {dm_time}")
