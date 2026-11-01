Posterior sampling of NZSHM22 rupture rates

The NZSHM22 inversion asks for the annual rate of every rupture in a rupture set such that the
predicted slip rates, the regional magnitude-frequency distribution (MFD) and the paleoseismic
event rates match their targets within their uncertainties: a linear system G m ≈ d with each row
already divided by its uncertainty. This workflow maps the whole set of models consistent with the data and a
stated prior: a Bayesian posterior, sampled by an exact Gibbs sampler, from which it reports
posterior means and spreads of every rupture rate, of every predicted datum and of sums such as
fault event rates, together with weighted representative models for hazard calculations.

Output figures from a run on a setup with 211,000 ruptures, very similar to the NZSHM rupture set:

<img width="2669" height="1398" alt="fig_slip_sections_ruptures_by_magnitude_1" src="https://github.com/user-attachments/assets/0f3dd5d8-755c-48e8-8365-04f4a772f013" />
<img width="2250" height="1398" alt="fig_slip_sections_ruptures_by_magnitude_2" src="https://github.com/user-attachments/assets/ec14095d-c772-423d-a76d-b5647650ab2c" />
<img width="1290" height="1282" alt="fig_rate_magnitude" src="https://github.com/user-attachments/assets/107682cf-06ce-40fb-acd6-816b65d1df1b" />
<img width="982" height="1376" alt="fig_paleo_rates" src="https://github.com/user-attachments/assets/4cbb0669-44f4-41a6-af2f-b025405813a7" />
<img width="1590" height="1179" alt="fig_slip_rates" src="https://github.com/user-attachments/assets/c8006852-cb80-422b-9447-0944ed0fd3bc" />
