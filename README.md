Posterior sampling of NZSHM22 rupture rates

The NZSHM22 inversion asks for the annual rate of every rupture in a rupture set such that the
predicted slip rates, the regional magnitude-frequency distribution (MFD) and the paleoseismic
event rates match their targets within their uncertainties: a linear system G m ≈ d with each row
already divided by its uncertainty. The NZSHM22 solutions find one such model by simulated
annealing. This workflow instead maps the whole set of models consistent with the data and a
stated prior: a Bayesian posterior, sampled by an exact Gibbs sampler, from which it reports
posterior means and spreads of every rupture rate, of every predicted datum and of sums such as
fault event rates, together with weighted representative models for hazard calculations.

Output figures from a run on a setup with 211,000 ruptures, very similar to the NZSHM rupture set, are below.

<img width="2669" height="1488" alt="fig_slip_sections_ruptures_by_magnitude_1" src="https://github.com/user-attachments/assets/dafd3763-5b6d-4b97-b275-067a74fece72" />
<img width="2250" height="1488" alt="fig_slip_sections_ruptures_by_magnitude_2" src="https://github.com/user-attachments/assets/7592aab1-9dfe-45c1-900f-69990dd31b2e" />
<img width="1290" height="1419" alt="fig_rate_magnitude" src="https://github.com/user-attachments/assets/0a03ca81-dc9a-445c-ab6b-98a572fd6338" />
<img width="1590" height="1345" alt="fig_slip_rates" src="https://github.com/user-attachments/assets/3ad136a0-8d73-49cc-a60c-bfc4d8812535" />
<img width="1590" height="1345" alt="fig_paleo_rates" src="https://github.com/user-attachments/assets/a3e7916e-b45f-42ef-8cd5-458672d13029" />
