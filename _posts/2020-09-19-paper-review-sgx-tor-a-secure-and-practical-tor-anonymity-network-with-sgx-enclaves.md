---
title: "[Paper review] - SGX-Tor: A Secure and Practical Tor Anonymity Network With
  SGX Enclaves"
layout: post
img: tee.png
---

*S. Kim, J. Han, J. Ha, T. Kim, and D. Han, "SGX-TOR: A secure and
practical TOR anonymity network with SGX enclaves", IEEE/ACM
Transactions on Networking, vol. 26,no. 5, pp. 2174--2187, 2018*

Context
-------

Intel SGX is a set of extensions to the Intel architecture that aims to
provide integrity and confidentiality guarantees to security sensitive
computation performed on a computer where all the privileged softwares
(kernel, hypervisor, etc) are potentially malicious$^1$.
It provides isolated execution by putting and executing the code and
data of an application inside a secure container called an *enclave*
$^2$. Indeed, by storing and computing information into
enclaves, Intel opened opportunities for a wide set of usages. Using
secure enclaves to protect sensitive information is of increasing
interest to the community of researchers who see in this method a new
approach to preserve anonymity of individuals, especially on cloud
servers. Hence the interest of the authors of this paper in the
implementation of this technology on the TOR network considered as being
amongst the best way of preserving the anonymity of online users.
Indeed, TOR is an anonymity network built on an architecture relying on
many remote layers used to anonymize connections and to preserve users'
privacy. Several attacks have been set up against TOR, some complex and
often effective. Attacks have diversified over time mostly targeting the
network layer, application layer and protocol layer.

Summary
-------

In the paper, Kim et al. present a novel approach of enhancing security
and privacy of TOR by using a Trusted Execution Environment (TEE) such
as Intel SGX. They make a focus on attacks and information leaks that
target TOR components. Indeed, as a volunteer-based network, TOR could
allow an attacker to add malicious components such as relays or to
compromise such components. The authors consider an adversary as the one
being able to modify or extract information from a TOR relay, compromise
hardware components such as memory and I/O devices except for the CPU
package itself. The authors also consider software components as
adversaries, this includes a privileged software such as the operating
system, hypervisor and BIOS. They present the security implication of
applying a TEE on TOR and evaluate the performance of such method. By
adding a new layer a security to TOR components, SGX-TOR prevents code
modification and limits the information exposed to untrusted parties at
the cost of a performance overhead. Indeed, as presented in the
evaluation section, this technique impacts TOR's performance by
increasing the end-to-end latency by 3.9% and the throughput overheads
for HTTP connections by 11.9%.

Contribution
------------

By offering anonymity to its users, the TOR network is the target of
many attacks. These attacks mainly target its components, especially
relays. Indeed, TOR becomes vulnerable when an attacker controls a large
fraction of relays. Hence the need to have a large and secure network.
In this publication, the authors focus on the security aspect of TOR
components by using a TEE, in particular Intel SGX which is increasingly
deployed on recent machines with Intel processors. This approach, widely
used for sensitive systems (such as location data), adds a layer of
security often at the expense of performance, this is the case with the
TOR network.
The authors apply this novel approach to increase user's anonymity
through the TOR network by thwarting some common attacks described in
the summary. Thwarting TOR's weaknesses to increase its level of privacy
is a great way to provide full anonymity online.

Limitation
----------

In this paper, the authors explored the potential offered by Intel SGX
in increasing the security of TOR components. However, their solution
shows some limitations that can be patched to increase the desirability
of the system or to extend the work to other areas of research.

1.  **Performance overhead**. The technology presented in the paper,
    although effective in some use cases, remains a nice to have.
    Indeed, for preserving anonymity, TOR uses relay networks run by
    volunteers without giving any guarantees of a high level of service.
    Indeed, one main drawback of the TOR network is its slowness due to
    its limited capacity and the increasing number of users
    $^3$. This matter of fact makes the solution
    very slow with basic usages. As presented in the evaluation section,
    by running SGX-TOR in a private environment, the results show that
    the latency increases by up to 18.4% and the performance reduces by
    25.6%. Adding a new layer of complexity affects the performance of
    the whole system and reduces the interest of the solution.

2.  **Conflicting statements**. In the scope definition of the paper
    (Section 3), the authors exclude Denial-of-Service (DoS) attacks and
    justify it by the ability of the system to simply deny the service
    (halt or reboot). Meanwhile, in Subsection 3.C, the authors list all
    the attacks thwarted by SGX-TOR and make a focus on snipper attack.
    A snipper attack, as presented by Loesing Karsten$^4$ is a destructive DoS attack that could disable
    TOR relays by making them to use an arbitrary amount of memory. This
    attack is based on TOR proxy (client) modification and can be
    thwarted by SGX-TOR when the client uses SGX.

3.  **Compatibility of Intel SGX with TOR components**. Another
    limitation of this paper is that it focuses on an approach where all
    the TOR main components (clients, directory servers and relays) are
    using SGX-enabled hardwares. As presented in the paper, SGX-TOR is
    only compatible with computers running Linux or Windows and having
    the hardware capabilities to use SGX. This excludes a large number
    of components, especially TOR Proxies. Regarding other components
    such as TOR relays, it is worth noting that these components are run
    by volunteers, TOR does not have a business model and reports a
    revenue mostly generated by donations and funding from the U.S.
    Government$^5$.

4.  **Intel SGX as an unbreakable solution**. Another limitation of the
    paper is that the authors present Intel SGX by considering it as
    being an unbreakable solution to preserve data integrity and
    privacy. The authors do not question Intel SGX itself. In the
    literature review, they do not mention nor consider attacks made on
    Intel SGX. In fact, many attacks such as enclave attacks or
    spectre-like attacks (Foreshadow attack)$^6$ have
    shown the weaknesses of Intel SGX. One interesting attack is a
    Prime+Probe type attack presented by Schwarz et al. in 2017 that can
    be used to grab RSA keys from SGX enclaves running on the same
    system by using CPU instructions in lieu of a fine-grained timer to
    exploit cache DRAM side-channels$^7$.
		
		
## 		References
		
1. Costan, V., & Devadas, S. (2016). Intel SGX Explained. IACR Cryptol. ePrint Arch., 2016(86), 1-118.
2. Kim, S., Han, J., Ha, J., Kim, T., & Han, D. (2018). SGX-Tor: A Secure and Practical Tor Anonymity Network With SGX Enclaves. IEEE/ACM Transactions on Networking, 26(5), 2174-2187.
3. http://www.torproject.org/press/presskit/2009-03-11-performance.pdf
4. Loesing, K. (2009). Privacy-enhancing technologies for private services (Vol. 2). University of Bamberg Press.
5. https://blog.torproject.org/transparency-openness-and-our-2016-and-2017-financials
6. Van Bulck, J., Minkin, M., Weisse, O., Genkin, D., Kasikci, B., Piessens, F., ... & Strackx, R. (2018). Foreshadow: Extracting the keys to the intel {SGX} kingdom with transient out-of-order execution. In 27th {USENIX} Security Symposium ({USENIX} Security 18) (pp. 991-1008).
7. Schwarz, M., Weiser, S., Gruss, D., Maurice, C., & Mangard, S. (2017, July). Malware guard extension: Using SGX to conceal cache attacks. In International Conference on Detection of Intrusions and Malware, and Vulnerability Assessment (pp. 3-24). Springer, Cham.
