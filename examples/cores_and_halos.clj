(ns cores-and-halos
  (:require
    [tech.v3.datatype.functional :as f]
    [tech.v3.datatype :as dtype]
    [tech.v3.tensor :as dtt]
    [tech.v3.datatype.bitmap :as bitmap]
    [fastmath.random :as fm.rand]
    [fastmath.core :as fm]
    [bennischwerdtner.sdm.sdm :as sdm]
    [bennischwerdtner.hd.binary-sparse-segmented :as hd]
    [bennischwerdtner.pyutils :as pyutils]
    [tech.v3.datatype.unary-pred :as unary-pred]
    [tech.v3.datatype.argops :as dtype-argops]
    [bennischwerdtner.hd.codebook-item-memory :as codebook]
    [bennischwerdtner.hd.ui.audio :as audio]
    [bennischwerdtner.hd.data :as hdd]
    [libpython-clj2.require :refer [require-python]]
    [libpython-clj2.python :refer [py. py..] :as py]))

(try (require-python '[numpy :as np])
     (require-python '[torch :as torch])
     (require-python '[torch.sparse :as torch.sparse])
     true
     (catch Exception e false))

(alter-var-root
 #'hdd/*item-memory*
 (constantly (codebook/codebook-item-memory 1000)))


;; --------------
;; Hebb 1949
;; Braitenberg 'thought pump'
;;
;;
;; --------------

(def sdm
  (sdm/->sdm {:address-count (long 1e6)
              :address-density 0.00003
              :word-length (long 1e4)}))

(def opts
  {:address-count (long 1e6)
   :address-density 0.00003
   :word-length (long 1e4)})

(def decoder (sdm/->decoder-coo opts))
(def storage (sdm/->sdm-storage-coo opts))

;; 1.
;; 'drop' is deterministic, so we can find a subset of the data
;;
;;

(def a (hdd/clj->vsa* :a))
(hd/drop a 0.5)

(py.. (torch/sum (sdm/decode-address decoder (hd/drop a 0.5) 1)) item)
320

(into
 [] (for [drop-rate (range 1 10)]
      (let [drop-rate (/ 1 drop-rate)]
        [drop-rate :address-count
         (py.. (torch/sum (sdm/decode-address
                           decoder
                           (hd/drop a drop-rate)
                           1))
           item)])))

'[[1 :address-count 0]
 [1/2 :address-count 320]
 [1/3 :address-count 459]
 [1/4 :address-count 491]
 [1/5 :address-count 518]
 [1/6 :address-count 574]
 [1/7 :address-count 596]
 [1/8 :address-count 596]
  [1/9 :address-count 596]]

;;
;; more address count is like asking for a bigger sub space of address space
;;
;;

;; 2.
;; Via sdm top-k, ask for more than one content basin in the attractor landscape of the address space
;; i.e. this  is similar to filling the available address space with more water and then looking where the water was flowing.
;; ~ If there are more than 1 basin, then top-k > 1 will fill both.
;; (otherwise will be roughly random)
;;


;;
;; One idea is that cores represent templates or frameworks
;; and halos represent the fillers.
;;

;;
;; What fits with this notion is that cores would be the intersection of multiple situations,
;;

;; Braitenberg 1977:

;; - this is a Hebbian autio-associative cell assembly definition:
;;
;; A cell assembly is a set of neurons interconnected by excitatory synapses.
;;
;; - each of the neurons receives excitation from and gives excitation to some other membmers of the
;;   same set
;; - cannot be separated into collecitons without severing ehat least two excitatory fibers,
;;   one for each direction.
;;
;;
;; Halo:
;; - Is by this definition not part of the assembly
;; - are neurons activated from, but not activating back to, the assembly
;;
;; - conversely, the *afferents* give it excitation but don't recieve from it
;;
;; - it is not neccesary that it is completely connected
;;
;; Sub-assemblies:
;; - a cell assmbly can contain sub-assemblies for which the same definition holds
;; - actually the entire nervous system, minus motor and sensory neurons (provided that they are not
;;   included in the feedback loops) is a cell assembly of which all the others are sub-assemblies.
;;
;;
;; So the Braitenberg 1977 assembly is an auto-associative strategic coalition, self-igniting subset of neurons
;; (which can be the complete set of neurons in the nervous system)

;;
;;
;;


;; Threshold control:
;; - threshold of excitation
;; - could be background excitation or inhibtion or true control of the threshold
;; (they mention a true control 'the least realistic assumption').
;; So they think more along the lines of circuits and cell phys. properties.
;;
;;
;; A cell assembly 'holds at threshold Θ'
;; - it can have sub assemblies that hold at lower or higher thresholds
;;
;;
;; improper cell assembly:
;; - collection of neurons with excitation, because each is part of a proper cell assembly,
;;   but not all of the same.
;; - they lack the fundamental property that partial excitation of the assembly spread to ingite the entire assembly.
;;

;; homogenous: each component gives and recieves excitation of a certain strenth to (from) the same number of other
;; neurons of the same assembly.
;;
;; non-homogenous: it may contain sub-assemblies wich are more strongy connected that the rest (*centers*)
;;
;; monocetric and polycentric non-homogenous cell assemblies; dependin on whether at higher threshold one or several disjunct sub-assemblies
;; will hold
;; Cell assemblies are disjunct when the do not share any neurons.
;;
;;
;;
;; Global inhibition:
;; - prevent epilepsy (maximal cell assembly)
;; - keep activity above a minimal level
;; - distinguish proper from improper cell assemblies (interesting, reader-centric assembly seems to throw this out)
;; - in order to discover centers within a cell assembly
;;



;;                       +------------------+
;;                       | A                |
;;                   +---+---+              |
;;     inputs        |       |   output     |
;;  ---------------->| B     +--------->    |
;;                   |       |              |
;;                   +----^--+              |
;;                        | Θ      +-----+  |
;;                        +--------+ Θ(A)<--+
;;                                 +-----+
;; A - activation
;; Θ - threshold
;; B - box containing neurons
;;


;; The threshold control mechanism  Θ(A):
;;-----------------------------------------------
;;

;; - the simplest definition of A is the number of active neurons within the box B
;; - they say it would be interesting if the threshold depends on the rate of A and theta
;; - in ensembles.clj Θ(A) is such that the activated neurons is constant (constantly capk)
;; - so it is a special case of this arrangement
;; - suppose theta is the same for all neurons


;; The next part in the paper would be cool to explore via software:
;;





;; -----------
;; FI (initial activated neurons from input)
;; EFI (the excited inputs)
;; F (I + EFI), generally different and doesn't even include the neurons of FI
;; F*I the final state ('the interpretation' of the inputs)

;; thought pump:
;; - Given I, lower the threshold so that the set of active neurons FI will go over into a larger set F'I
;; - This will encourage the ignition of cell assemblies
;; - As the threshold is again raised, activity is smothered and only the most strongly connected cell assembly will survive
;; - A  new cycle beginning again witht a lowerd threshold will bring in new cell assemblies.
;; - They may include an even more strongly connected cell assembly, which will be the nex one to survive when the threshold is raised
;; - the evoution will be in the direction of the most strongly connected cell assemblies.
;; - more interesting with adapation or fatigue (makes the persisting ones fade and moves it to the next)
;;


;; Temporal structure
;; - they come with syllables via periodicity in threshold control.
;; - (this is exactly what Buzsáki also says with the cipher / neuronal syntax)
;; -
;;

;; - they say how the assemblies can represent things, if the neurons are attributes or features
;; - this is actually the "calculus of ideas" from McCulloch and Pitts 1943
;;

;; - the thought pump or sequences they say fits with 'chain of associations' in psychology
;;
;; - the halo represents the `consequences` of the `thing` represented, and will determine the next cell assembly that will
;; ignite in the sequence 👈
;;
;; (this fits with Buzsáki's sequence view, too I think)
;;
;; - so the idea that forward connections work hetero-associative
;; - and also the idea that 'moving the ideas forward' can be done by increasing the excitability,
;; - but at Buzsáki the sequences move on their own, they are sped up by movement speed instead
;;
;; - connections within a cell assembly: 'the symmetrical relation of belonging together' (auto associative)
;; - while the second (the ones going from center to halo) embody the asymmetrical relation of consequence, temporal sequence or causality (hetero associative)
;;








;; ---------------------
;;
;;
;; Difference to the reader centric assembly:
;; - reader centric assembly puts the *effect* at the center, this is a more powerful concept,
;;   else the question remains what are the assemblies doing?
;; - reader centric assembly makes the role of time windows clear, it's the integration time of the reader.
;; - reader centric assembly is not necessarily auto-associative. It's more general.
;; - the reader centric assembly is distributed datastructure, but it is neither fully distributed nor local,
;;   it's distributed across all the potential inputs to a reader I suppose.
;; - the improper cell assembly is explicitly proper for the reader centric assembly. Cells need to share a reader, theiri
;;   internal connectivity doesn't matter for the reader centric definition.
;;   This is the exact oposite of the self-igniting view, where it's the internal structure and not the readers that matter.
;; - for reader centric, one could define disjunct by saying they they don't share a reader
;;   there is a nuance: The potential reader or the activated reader?
;;
;;
;;
;;
;;
;; - my feel is the reader centric view makes it more useful in an information processing view (i.e. what is the message?)
;; - while the self-igniting view makes it more intuitive in a memes eye view (i.e. how does it survive?)
;;
;;
;;
;; - With an sdm we can auto-associate address and content, this I think captures the auto association.
;;
;;
;; --------------------


;; 'ignite' comes from Charles Legéndy:
;; https://en.wikipedia.org/wiki/Charles_Leg%C3%A9ndy
