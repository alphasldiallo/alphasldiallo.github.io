---
title: "[Paper Review] Don’t trust the locals: Investigating the prevalence of persistent
  client-side cross-site scripting in the wild"
layout: post
img: xss.png
tags:
- XSS
- Security
- Storage
---

*M. Steffens, C. Rossow, M. Johns, and B. Stock, “Don’t trust the locals: Investigating the prevalence of persistent client-side cross-site scripting in the wild.,” 2019*

## Context
Cross site scripting (XSS) can be defined as a security vulnerability found on web applications which consists of injecting client-side scripts to an application. It is widely spread and considered as being the most nefarious attack against Web clients. The research community has long focused on three main types of XSS attacks: Reflected, persistent and DOM-based XSS . The detection and mitigation of these attacks have received a lot of attention unlike the persistent client-side XSS presented by the authors. To raise awareness on this new type of XSS attacks, the authors studied its prevalence and exploited some flaws in some of the top 5,000 Alexa domains.
## Summary
In this paper, Steffens et al. present the first systematic study on the threat of Persistent client-side XSS mostly enabled by the emergence of persistence APIs on the client side. The paper shows the threat of such persistent XSS on local storage by identifying vulnerabilities in the Alexa top 5,000 websites. The authors implement two attacker models capable of injecting malicious payloads into storage. The first attacker model leverages the network to temporarily hijacking a non-encrypted connection meanwhile the second forces the victims to visit arbitrary URLs.

By leveraging taint tracking to identify suspicious flows from client-side persistent storage to sinks and by using the two attacker models listed above, they found that more than 8% of the Alexa top 5,000 domains have unfiltered data flows from persistent storage to a dangerous sink. By considering only sites that make use of data originating from local storage, they found out that 21% of the sites are vulnerable to client-side persistent XSS and in this subset, 70% are directly exploitable by the two attacker models.
## Contribution
The opportunity offered by local storage opens the door to numerous attacks such as the one presented in the paper written by Steffens et al. Indeed, the dangers of insecure client-side usage of stored code and data under potential control of an adversary have not been studied systematically. The main contribution of the authors is to raise awareness on a security breach affecting modern browsers. By studying the prevalence of vulnerabilities opening breaches to a client-side persistent XSS attack and exploiting these breaches, the authors draw the attention of the industry and the research community on the damages that can be caused by such attack. Unlike reflected and persistent (server-side) attacks, a client-side persistent XSS flaw is difficult to spot by the developers. In spite of the pernicious nature of this security flaw, it is not very well documented compared to the flaws allowing reflected XSS attacks.
## Limitation
The attack presented in this paper shows relevant flaws to trigger attention of developers, industries and researchers. However, this paper has some limitations that could be patched as mentioned below.

**Extending the sample.** To measure the prevalence of client-side XSS as presented in the paper, the authors focus on the most visited websites based on the Alexa Ranking. The authors detected only 8% of this subset which exhibit exploitable flaw from client-side storage. In order to extend their study and have a better idea of the prevalence of such flaw in the wild, the authors could had investigated the vulnerabilities on a larger sample of websites by exploiting the fact that 43.6% of websites use custom-made Content Management System (CMS), mostly Wordpress (34%). Since most CMS are built and installed with the default configuration, a security flaw detected on one version of a CMS or on a popular plugin can affect a large number of websites.

**Category of a client-side persistent XSS attack.** In this paper, the authors defend the fact that a client-side XSS attack should be considered as a new category of XSS attacks. However, by taking a closer look at the definition of Persistent XSS as presented by the Web Application Security Consortium (WASC) , we realize that it has almost the same principles. Although, Persistent Server Side XSS appears to be even more devastating as it may affect a wider range of users as seen with the famous hack of the Ubuntu forum in 2013 . To avoid any ambiguity, it would be more appropriate to consider the flaw presented in the paper as a subset of Persistent XSS.


*Notes :*

The article was reviewed and presented during my reading exam at the University of Lausanne in May 22, 2020.

The illustration image is AI generated.
