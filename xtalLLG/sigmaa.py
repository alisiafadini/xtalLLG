import torch
from xtalLLG import targets, dsutils, structurefactors
from tqdm import tqdm
import numpy as np


def refine_sigmaa(unique_labels, bin_labels, Eobs, Ecalc, centric):
    # Training loop
    num_epochs = 25
    sigma_As = [[] for _ in range(len(unique_labels))]
    corr_coefs = [torch.tensor(0.0, dtype=torch.float32) for _ in unique_labels]

    for i, label in tqdm(enumerate(unique_labels)):
        bin_indices = bin_labels == label
        bin_Eobs = Eobs[bin_indices]
        bin_Ecalc = Ecalc[bin_indices]

        # initialize sigmaA values with correlation coefficient
        corr_coefs[i] = torch.corrcoef(torch.stack((bin_Eobs, bin_Ecalc), dim=0))[1][0]
        corr_coefs[i] = torch.clamp(corr_coefs[i], min=0.001, max=0.999)
        print("correlation coeff", corr_coefs[i])
        sigma_As[i] = np.sqrt(corr_coefs[i].item())

        sigma_As[i] = torch.tensor(
            sigma_As[i],
            dtype=torch.float32,
            requires_grad=True,
            device=dsutils.try_gpu(),
        )

        # optimizer = torch.optim.RMSprop([sigma_As[i]], lr=6e-4)
        optimizer = torch.optim.Adam([sigma_As[i]], lr=1e-3)

        for epoch in range(num_epochs):
            optimizer.zero_grad()  # Clear gradients
            sigma_A = sigma_As[i]

            # Compute LLG expression for the bin
            llg = targets.llgTot_calculate(
                sigma_A, bin_Eobs, bin_Ecalc, centric[bin_indices]
            )

            # Minimize the negative LLG (maximize LLG)
            loss = -llg
            loss.backward(retain_graph=True)

            # Update the current sigma_A
            # optimizers[i].step()
            optimizer.step()

            # Enforce SIGMAA bounds
            sigma_A.data = torch.clamp(sigma_A.data, 0.015, 0.99)

    return sigma_As


def sigmaA_from_model(
    E_true, phi_true, sfcalculator_model, eps, sigmaN_model, bin_labels
):
    phitrue_rad = np.deg2rad(phi_true)

    E_model = structurefactors.normalize_Fs(
        structurefactors.ftotal_amplitudes(sfcalculator_model, "Ftotal_HKL"),
        eps,
        sigmaN_model,
        bin_labels,
    )

    phimodel = dsutils.assert_numpy(
        structurefactors.ftotal_phis(sfcalculator_model, "Ftotal_HKL")
    )
    phimodel_rad = np.deg2rad(phimodel)

    sigmaAs = structurefactors.compute_sigmaA_true(
        dsutils.assert_numpy(E_true),
        phitrue_rad,
        dsutils.assert_numpy(E_model),
        phimodel_rad,
        dsutils.assert_numpy(bin_labels),
    )

    return sigmaAs
