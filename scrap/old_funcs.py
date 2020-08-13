def calcTwoD(n_loopsum):#
    ''' Caculates a 2d spectrum for both perpendicular and parael lasers
    using aceto bands and accurate lineshapes'''

    container = []

    #energies0 = [energy - (100 * num_mol / 2) + i * 100 for i in range(num_mol)]
    energies0 = [energy] * num_mol
    # Giving random energies to the moleucles according to a gauss dist
    with qr.energy_units("1/cm"):
        for i, mol in enumerate(for_agg):
            #mol.set_energy(1, random.gauss(energy, static_dis))
            mol.set_energy(1, energies0[i])

    agg = qr.Aggregate(molecules=for_agg)
    agg.set_coupling_by_dipole_dipole(epsr=1.21)
    agg.build(mult=2)
    agg.diagonalize()
    rwa = agg.get_RWA_suggestion()

    # Initialising the twod response calculator for the paralell laser
    resp_cal_para = qr.TwoDResponseCalculator(
        t1axis=t13_ax,
        t2axis=t2_ax,
        t3axis=t13_ax,
        system=agg
        )
    # Copying the response calculator for the perpendicular laser
    resp_cal_perp = resp_cal_para

    # Bootstrap is the place to add 0-padding to the response signal
    # printEigen=True prints eigenvalues, printResp='string' prints response
    # Response is calculated Converted into spectrum Stored in a container
    resp_cal_para.bootstrap(rwa, pad=padding, lab=lab_para)
    resp_para_cont = resp_cal_para.calculate()
    spec_cont_para = resp_para_cont.get_TwoDSpectrumContainer()
    container.append(spec_cont_para)

    # REpeating the above process for the perpendicular laser setup
    resp_cal_perp.bootstrap(
        rwa,
        pad=padding,
        verbose=True,
        lab=lab_perp,
        printEigen=False
        )#, printResp='perpResp'
    resp_perp_cont = resp_cal_perp.calculate()
    spec_cont_perp = resp_perp_cont.get_TwoDSpectrumContainer()
    container.append(spec_cont_perp)

    #return container
    return spec_cont_para, spec_cont_perp
