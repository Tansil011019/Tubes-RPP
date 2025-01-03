(deftemplate city
  (slot name)
  (slot total-score)
  (slot cost)
  (slot quality-of-life)
  (slot internet)
  (slot safety)
  (slot fun)
  (slot walkability)
  (slot nightlife)
  (slot friendly-to-foreigners)
  (slot freedom-of-speech)
  (slot english-speaking)
  (slot food-safety)
  (slot places-to-work-from))

(deftemplate user-preferences
  (slot cost-weight)
  (slot safety-weight)
  (slot fun-weight)
  (slot quality-of-life-weight)
  (slot internet-weight)
  (slot nightlife-weight)
  (slot english-speaking-weight)
  (slot food-safety-weight)
  (slot freedom-of-speech-weight)
  (slot places-to-work-from-weight))

(deftemplate city-score
  (slot name)
  (slot score))

(defrule collect-user-input
  => 
  (printout t "Enter your weight for Cost (1-10): " crlf)
  (bind ?cost (read))
  (printout t "Enter your weight for Safety (1-10): " crlf)
  (bind ?safety (read))
  (printout t "Enter your weight for Fun (1-10): " crlf)
  (bind ?fun (read))
  (printout t "Enter your weight for Quality of Life (1-10): " crlf)
  (bind ?quality-of-life (read))
  (printout t "Enter your weight for Internet (1-10): " crlf)
  (bind ?internet (read))
  (printout t "Enter your weight for Nightlife (1-10): " crlf)
  (bind ?nightlife (read))
  (printout t "Enter your weight for English Speaking (1-10): " crlf)
  (bind ?english-speaking (read))
  (printout t "Enter your weight for Food Safety (1-10): " crlf)
  (bind ?food-safety (read))
  (printout t "Enter your weight for Freedom of Speech (1-10): " crlf)
  (bind ?freedom-of-speech (read))
  (printout t "Enter your weight for Place to Work (1-10): " crlf)
  (bind ?place-to-work-from (read))
  (assert (user-preferences (cost-weight ?cost)
                            (safety-weight ?safety)
                            (fun-weight ?fun)
                            (quality-of-life-weight ?quality-of-life)
                            (internet-weight ?internet)
                            (nightlife-weight ?nightlife)
                            (english-speaking-weight ?english-speaking)
                            (food-safety-weight ?food-safety)
                            (freedom-of-speech-weight ?freedom-of-speech)
                            (places-to-work-from-weight ?place-to-work-from))))

(defrule calculate-score
   (user-preferences (cost-weight ?cost-w) (safety-weight ?safety-w)
                     (fun-weight ?fun-w) (quality-of-life-weight ?quality-w)
                     (internet-weight ?internet-w) (nightlife-weight ?nightlife-w)
                     (english-speaking-weight ?english-speaking-w)
                     (food-safety-weight ?food-safety-w)
                     (freedom-of-speech-weight ?freedom-w)
                     (places-to-work-from-weight ?places-w))
   (city (name ?name) (total-score ?total-score) (cost ?cost)
         (quality-of-life ?quality-of-life) (internet ?internet) (safety ?safety)
         (fun ?fun) (nightlife ?nightlife) (english-speaking ?english-speaking)
         (food-safety ?food-safety) (freedom-of-speech ?freedom)
         (places-to-work-from ?places))
   =>
   ;; Calculate the Euclidean distance
   (bind ?distance (sqrt (+ (** (- ?cost-w ?cost) 2)  ; Distance between user weight and cost
                            (** (- ?safety-w ?safety) 2)
                            (** (- ?fun-w ?fun) 2)
                            (** (- ?quality-w ?quality-of-life) 2)
                            (** (- ?internet-w ?internet) 2)
                            (** (- ?nightlife-w ?nightlife) 2)
                            (** (- ?english-speaking-w ?english-speaking) 2)
                            (** (- ?food-safety-w ?food-safety) 2)
                            (** (- ?freedom-w ?freedom) 2)
                            (** (- ?places-w ?places) 2))))

   ;; Define the maximum distance (arbitrarily large to scale percentage properly)
   (bind ?max-distance 100)

   ;; Calculate the percentage score
   (bind ?percentage (- 100 (/ (* ?distance 100) ?max-distance)))

   ;; Assert the percentage score
   (assert (city-score (name ?name) (score ?percentage)))
   ;; Print the percentage for the city
   (printout t "Calculated score for " ?name ": " ?percentage "%" " (Distance: " ?distance ")" crlf))


(deffunction my-predicate (?city-score1 ?city-score2)
  (> (fact-slot-value ?city-score1 score) (fact-slot-value ?city-score2 score))
)

(deffunction find-max (?template ?predicate)
  (bind ?max FALSE)
  (do-for-all-facts ((?f ?template)) TRUE
    (if (or (not ?max) (funcall ?predicate ?f ?max))
      then
      (bind ?max ?f)
    ))
  (return ?max)
)

(defrule find-max
  =>
  (bind ?city-score (find-max city-score my-predicate))
  (if ?city-score
    then
   (printout t "Final recommended city: " (fact-slot-value ?city-score name) " with score: " (fact-slot-value ?city-score score) crlf)
  )
)

; Added normalized city data
(deffacts city-data
	(city (name "Bangkok") (total-score 10.00) (cost 2.59) (quality-of-life 10.00) (internet 3.50) (safety 8.20) (fun 7.00) (walkability 10.00) (nightlife 10.00) (friendly-to-foreigners 10.00) (freedom-of-speech 10.00) (english-speaking 1.00) (food-safety 10.00) (places-to-work-from 10.00))
	(city (name "Chiang-Mai") (total-score 8.76) (cost 1.66) (quality-of-life 10.00) (internet 3.79) (safety 8.20) (fun 4.00) (walkability 10.00) (nightlife 1.00) (friendly-to-foreigners 6.40) (freedom-of-speech 10.00) (english-speaking 1.00) (food-safety 1.00) (places-to-work-from 10.00))
	(city (name "Kuala-Lumpur") (total-score 7.03) (cost 2.21) (quality-of-life 5.50) (internet 5.50) (safety 8.20) (fun 1.00) (walkability 4.00) (nightlife 1.00) (friendly-to-foreigners 6.40) (freedom-of-speech 10.00) (english-speaking 4.00) (food-safety 10.00) (places-to-work-from 10.00))
	(city (name "Singapore") (total-score 6.69) (cost 10.00) (quality-of-life 10.00) (internet 3.79) (safety 10.00) (fun 7.00) (walkability 10.00) (nightlife 4.00) (friendly-to-foreigners 6.40) (freedom-of-speech 10.00) (english-speaking 10.00) (food-safety 10.00) (places-to-work-from 10.00))
	(city (name "George-Town") (total-score 6.66) (cost 1.35) (quality-of-life 10.00) (internet 10.00) (safety 8.20) (fun 4.00) (walkability 10.00) (nightlife 7.00) (friendly-to-foreigners 10.00) (freedom-of-speech 10.00) (english-speaking 7.00) (food-safety 10.00) (places-to-work-from 7.00))
	(city (name "Makassar") (total-score 6.66) (cost 1.09) (quality-of-life 5.50) (internet 1.50) (safety 10.00) (fun 4.00) (walkability 10.00) (nightlife 4.00) (friendly-to-foreigners 6.40) (freedom-of-speech 10.00) (english-speaking 1.00) (food-safety 5.50) (places-to-work-from 10.00))
	(city (name "Ubud") (total-score 6.62) (cost 3.56) (quality-of-life 10.00) (internet 2.50) (safety 10.00) (fun 10.00) (walkability 10.00) (nightlife 4.00) (friendly-to-foreigners 8.20) (freedom-of-speech 10.00) (english-speaking 1.00) (food-safety 5.50) (places-to-work-from 10.00))
	(city (name "Ipoh") (total-score 6.52) (cost 2.57) (quality-of-life 5.50) (internet 1.79) (safety 8.20) (fun 4.00) (walkability 10.00) (nightlife 1.00) (friendly-to-foreigners 6.40) (freedom-of-speech 10.00) (english-speaking 4.00) (food-safety 10.00) (places-to-work-from 10.00))
	(city (name "Medan") (total-score 6.50) (cost 1.04) (quality-of-life 5.50) (internet 1.50) (safety 10.00) (fun 4.00) (walkability 1.00) (nightlife 4.00) (friendly-to-foreigners 6.40) (freedom-of-speech 10.00) (english-speaking 1.00) (food-safety 5.50) (places-to-work-from 10.00))
	(city (name "Hanoi") (total-score 6.46) (cost 1.57) (quality-of-life 5.50) (internet 2.86) (safety 8.20) (fun 1.00) (walkability 10.00) (nightlife 4.00) (friendly-to-foreigners 6.40) (freedom-of-speech 7.00) (english-speaking 1.00) (food-safety 10.00) (places-to-work-from 10.00))
	(city (name "Da-Nang") (total-score 6.43) (cost 1.47) (quality-of-life 5.50) (internet 2.21) (safety 8.20) (fun 4.00) (walkability 10.00) (nightlife 4.00) (friendly-to-foreigners 8.20) (freedom-of-speech 7.00) (english-speaking 1.00) (food-safety 10.00) (places-to-work-from 10.00))
	(city (name "Cebu") (total-score 6.32) (cost 2.37) (quality-of-life 5.50) (internet 1.43) (safety 8.20) (fun 7.00) (walkability 10.00) (nightlife 4.00) (friendly-to-foreigners 8.20) (freedom-of-speech 10.00) (english-speaking 4.00) (food-safety 10.00) (places-to-work-from 10.00))
	(city (name "Ho-Chi-Minh-City") (total-score 6.29) (cost 1.43) (quality-of-life 5.50) (internet 2.14) (safety 8.20) (fun 7.00) (walkability 10.00) (nightlife 7.00) (friendly-to-foreigners 6.40) (freedom-of-speech 7.00) (english-speaking 1.00) (food-safety 10.00) (places-to-work-from 10.00))
	(city (name "Phnom-Penh") (total-score 5.95) (cost 1.75) (quality-of-life 5.50) (internet 1.29) (safety 8.20) (fun 4.00) (walkability 4.00) (nightlife 1.00) (friendly-to-foreigners 8.20) (freedom-of-speech 10.00) (english-speaking 1.00) (food-safety 5.50) (places-to-work-from 10.00))
	(city (name "Davao") (total-score 5.95) (cost 1.93) (quality-of-life 5.50) (internet 1.43) (safety 8.20) (fun 7.00) (walkability 10.00) (nightlife 4.00) (friendly-to-foreigners 10.00) (freedom-of-speech 10.00) (english-speaking 7.00) (food-safety 10.00) (places-to-work-from 10.00))
	(city (name "Jakarta") (total-score 5.63) (cost 2.03) (quality-of-life 5.50) (internet 1.50) (safety 10.00) (fun 7.00) (walkability 10.00) (nightlife 4.00) (friendly-to-foreigners 6.40) (freedom-of-speech 10.00) (english-speaking 4.00) (food-safety 5.50) (places-to-work-from 10.00))
	(city (name "Pattaya") (total-score 5.42) (cost 2.78) (quality-of-life 5.50) (internet 2.50) (safety 8.20) (fun 7.00) (walkability 10.00) (nightlife 7.00) (friendly-to-foreigners 10.00) (freedom-of-speech 10.00) (english-speaking 1.00) (food-safety 1.00) (places-to-work-from 10.00))
	(city (name "Bandung") (total-score 5.26) (cost 1.48) (quality-of-life 5.50) (internet 1.50) (safety 10.00) (fun 4.00) (walkability 10.00) (nightlife 1.00) (friendly-to-foreigners 8.20) (freedom-of-speech 10.00) (english-speaking 1.00) (food-safety 5.50) (places-to-work-from 10.00))
	(city (name "Surabaya") (total-score 5.21) (cost 1.13) (quality-of-life 5.50) (internet 1.43) (safety 10.00) (fun 1.00) (walkability 10.00) (nightlife 1.00) (friendly-to-foreigners 6.40) (freedom-of-speech 10.00) (english-speaking 1.00) (food-safety 5.50) (places-to-work-from 10.00))
	(city (name "Baguio") (total-score 4.91) (cost 3.21) (quality-of-life 5.50) (internet 1.43) (safety 8.20) (fun 1.00) (walkability 10.00) (nightlife 4.00) (friendly-to-foreigners 1.00) (freedom-of-speech 10.00) (english-speaking 7.00) (food-safety 10.00) (places-to-work-from 1.00))
	(city (name "Manila") (total-score 4.75) (cost 2.55) (quality-of-life 5.50) (internet 1.43) (safety 6.40) (fun 1.00) (walkability 10.00) (nightlife 7.00) (friendly-to-foreigners 6.40) (freedom-of-speech 10.00) (english-speaking 7.00) (food-safety 10.00) (places-to-work-from 10.00))
	(city (name "Vientiane") (total-score 4.11) (cost 1.00) (quality-of-life 5.50) (internet 1.00) (safety 8.20) (fun 1.00) (walkability 10.00) (nightlife 4.00) (friendly-to-foreigners 10.00) (freedom-of-speech 7.00) (english-speaking 1.00) (food-safety 5.50) (places-to-work-from 10.00))
	(city (name "Yangon") (total-score 1.00) (cost 1.53) (quality-of-life 1.00) (internet 1.21) (safety 1.00) (fun 10.00) (walkability 4.00) (nightlife 1.00) (friendly-to-foreigners 8.20) (freedom-of-speech 1.00) (english-speaking 1.00) (food-safety 10.00) (places-to-work-from 10.00))
	(city (name "Naypyidaw") (total-score 1.00) (cost 2.47) (quality-of-life 1.00) (internet 1.14) (safety 1.00) (fun 1.00) (walkability 10.00) (nightlife 1.00) (friendly-to-foreigners 4.60) (freedom-of-speech 1.00) (english-speaking 1.00) (food-safety 10.00) (places-to-work-from 7.00))
)
