% ==============================================================
% DOMAIN
% ==============================================================

clothing(well_dressed_people; casual_clothing; mixed_clothing).
language(formal_language; casual_language; insults).
voice(calm_tone; raised_voices; very_raised_voices).
noise(low_noise; normal_noise; high_noise).
personal_space(large_interpersonal_distance; close_distance).
faces(low_smiles; smiles; laughter).
posture(chill_posture; rigid_posture; aggressive_posture).

environment(neutral; hostile; formal; friendly; tense).

% ==============================================================
% REQUIRED OBSERVATIONS
% ==============================================================

required(formal, clothing(well_dressed_people)).
required(formal, language(formal_language)).

required(friendly, personal_space(close_distance)).
required(friendly, language(casual_language)).
required(friendly, voice(calm_tone)).

required(hostile, language(insults)).
required(hostile, voice(very_raised_voices)).

required(tense, voice(raised_voices)).

% ==============================================================
% INVALID ENVIRONMENTS
% ==============================================================

invalid(E) :-
    required(E,X),
    not observed(X).

% ==============================================================
% SCORE DEFINITIONS
% ==============================================================

% ================= FORMAL =================

score(formal, clothing(well_dressed_people),3).
score(formal, clothing(mixed_clothing),1).
score(formal, clothing(casual_clothing),0).

score(formal, language(formal_language),3).
score(formal, language(casual_language),1).
score(formal, language(insults),-2).

score(formal, voice(calm_tone),2).
score(formal, voice(raised_voices),-1).
score(formal, voice(very_raised_voices),-3).

score(formal, noise(low_noise),2).
score(formal, noise(normal_noise),1).
score(formal, noise(high_noise),-2).

score(formal, personal_space(large_interpersonal_distance),2).
score(formal, personal_space(close_distance),1).

score(formal, faces(low_smiles),2).
score(formal, faces(smiles),1).
score(formal, faces(laughter),-1).

score(formal, posture(rigid_posture),2).
score(formal, posture(chill_posture),0).
score(formal, posture(aggressive_posture),-3).


% ================= FRIENDLY =================

score(friendly, clothing(casual_clothing),2).
score(friendly, clothing(mixed_clothing),1).
score(friendly, clothing(well_dressed_people),0).

score(friendly, language(casual_language),3).
score(friendly, language(formal_language),1).
score(friendly, language(insults),-2).

score(friendly, voice(calm_tone),2).
score(friendly, voice(raised_voices),1).
score(friendly, voice(very_raised_voices),-3).

score(friendly, noise(low_noise),1).
score(friendly, noise(normal_noise),2).
score(friendly, noise(high_noise),2).

score(friendly, personal_space(close_distance),3).
score(friendly, personal_space(large_interpersonal_distance),0).

score(friendly, faces(smiles),2).
score(friendly, faces(laughter),3).
score(friendly, faces(low_smiles),0).

score(friendly, posture(chill_posture),2).
score(friendly, posture(rigid_posture),0).
score(friendly, posture(aggressive_posture),-3).


% ================= HOSTILE =================

score(hostile, clothing(casual_clothing),1).
score(hostile, clothing(mixed_clothing),1).
score(hostile, clothing(well_dressed_people),1).

score(hostile, language(insults),4).
score(hostile, language(casual_language),1).
score(hostile, language(formal_language),0).

score(hostile, voice(very_raised_voices),3).
score(hostile, voice(raised_voices),2).
score(hostile, voice(calm_tone),-2).

score(hostile, noise(high_noise),3).
score(hostile, noise(normal_noise),1).
score(hostile, noise(low_noise),0).

score(hostile, personal_space(close_distance),2).
score(hostile, personal_space(large_interpersonal_distance),0).

score(hostile, faces(low_smiles),2).
score(hostile, faces(smiles),-1).
score(hostile, faces(laughter),0).

score(hostile, posture(aggressive_posture),4).
score(hostile, posture(rigid_posture),2).
score(hostile, posture(chill_posture),-2).


% ================= TENSE =================

score(tense, clothing(mixed_clothing),1).
score(tense, clothing(well_dressed_people),1).
score(tense, clothing(casual_clothing),1).

score(tense, language(formal_language),1).
score(tense, language(casual_language),1).
score(tense, language(insults),2).

score(tense, voice(raised_voices),3).
score(tense, voice(very_raised_voices),2).
score(tense, voice(calm_tone),-1).

score(tense, noise(high_noise),3).
score(tense, noise(normal_noise),2).
score(tense, noise(low_noise),0).

score(tense, personal_space(large_interpersonal_distance),1).
score(tense, personal_space(close_distance),1).

score(tense, faces(low_smiles),2).
score(tense, faces(smiles),0).
score(tense, faces(laughter),-2).

score(tense, posture(rigid_posture),3).
score(tense, posture(aggressive_posture),2).
score(tense, posture(chill_posture),-1).


% ================= NEUTRAL =================

score(neutral, clothing(mixed_clothing),2).
score(neutral, clothing(well_dressed_people),1).
score(neutral, clothing(casual_clothing),1).

score(neutral, language(formal_language),1).
score(neutral, language(casual_language),1).
score(neutral, language(insults),-2).

score(neutral, voice(calm_tone),2).
score(neutral, voice(raised_voices),0).
score(neutral, voice(very_raised_voices),-2).

score(neutral, noise(normal_noise),2).
score(neutral, noise(low_noise),1).
score(neutral, noise(high_noise),-1).

score(neutral, personal_space(large_interpersonal_distance),1).
score(neutral, personal_space(close_distance),1).

score(neutral, faces(low_smiles),2).
score(neutral, faces(smiles),1).
score(neutral, faces(laughter),0).

score(neutral, posture(chill_posture),2).
score(neutral, posture(rigid_posture),1).
score(neutral, posture(aggressive_posture),-2).


% ==============================================================
% TOTAL SCORE
% ==============================================================

total_score(E,Total) :-
    environment(E),
    not invalid(E),
    Total = #sum { S,X : observed(X), score(E,X,S) }.

% ==============================================================
% POSSIBLE ENVIRONMENTS (ONLY POSITIVE SCORE)
% ==============================================================

possible(E) :-
    total_score(E,T),
    T > 0.

% ==============================================================
% CONTRIBUTIONS (FOR EXPLANATION)
% ==============================================================

contribution(E,X,S) :-
    total_score(E,_),
    observed(X),
    score(E,X,S).

positive(E,X,S) :-
    contribution(E,X,S),
    S > 0.

negative(E,X,S) :-
    contribution(E,X,S),
    S < 0.

% ==============================================================
% BEST ENVIRONMENT
% ==============================================================

best_score(M) :-
    M = #max { T,E : total_score(E,T), T > 0 }.

best(E) :-
    total_score(E,T),
    best_score(T).

% ==============================================================
% OUTPUT
% ==============================================================

#show possible/1.
#show total_score/2.
#show best/1.