# Cara
Cara approximates the optimal revenue of continuously-valued, additive, sealed-bid, multi-item auctions. 

The program takes as input an auction setting and desired parameters for approximation. Cara then tries to lower bound the optimal revenue in two ways:
1. Cara randomly takes a number of finite-support item valuation distributions that are stochastically-dominated by the original distribution and solves for their optimal revenues.
2. Cara considers the revenue of two specific second-priced auctions, namely bundling all items together in one second-price auction as well as selling the items separately in m different second-price auctions.

A formal overview of the auctions structure is given in Chapter 2 of `thesis.pdf` and detailed instructions for using Cara are found in Chapter 4. An overview of the codebase is given in Chapter 6. (Download the PDF to view it as the Github viewer does not work properly.)