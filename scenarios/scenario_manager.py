from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import random

# Predefined agent configurations
AGENT_CONFIGS = {
    "Dr. Sarah Chen": {
        "role": "doctor",
        "personality": {
            "openness": 0.8,
            "conscientiousness": 0.9,
            "extraversion": 0.6,
            "agreeableness": 0.8,
            "neuroticism": 0.3
        },
        "background": "Dr. Sarah Chen is a compassionate emergency physician with 15 years of experience. She believes in evidence-based medicine and always puts patient care first. She's worked in conflict zones and has seen the worst of humanity, making her both empathetic and realistic.",
        "goals": ["Ensure everyone's well-being", "Provide medical expertise", "Advocate for ethical decisions"]
    },
    "Marcus Steel": {
        "role": "engineer",
        "personality": {
            "openness": 0.7,
            "conscientiousness": 0.8,
            "extraversion": 0.4,
            "agreeableness": 0.6,
            "neuroticism": 0.4
        },
        "background": "Marcus Steel is a systems engineer who thinks in terms of efficiency and practical solutions. He has built infrastructure in developing countries and believes technology can solve most problems. He's logical, methodical, and sometimes struggles with emotional nuances.",
        "goals": ["Find practical solutions", "Optimize systems and processes", "Ensure technical feasibility"]
    },
    "Agent X": {
        "role": "spy",
        "personality": {
            "openness": 0.6,
            "conscientiousness": 0.7,
            "extraversion": 0.3,
            "agreeableness": 0.4,
            "neuroticism": 0.5
        },
        "background": "Agent X is a intelligence operative with a mysterious past. They excel at reading people and situations, always thinking three steps ahead. Their loyalty is to their mission, and they trust very few people completely.",
        "goals": ["Gather intelligence", "Protect classified information", "Complete the mission"]
    },
    "Maya Rodriguez": {
        "role": "rebel",
        "personality": {
            "openness": 0.9,
            "conscientiousness": 0.5,
            "extraversion": 0.8,
            "agreeableness": 0.4,
            "neuroticism": 0.6
        },
        "background": "Maya Rodriguez is a passionate activist who fights for social justice and environmental causes. She's not afraid to challenge authority and speak truth to power. Her idealism sometimes clashes with practical constraints, but her heart is always in the right place.",
        "goals": ["Fight for justice", "Challenge the status quo", "Give voice to the voiceless"]
    },
    "Ambassador Williams": {
        "role": "diplomat",
        "personality": {
            "openness": 0.7,
            "conscientiousness": 0.8,
            "extraversion": 0.7,
            "agreeableness": 0.9,
            "neuroticism": 0.2
        },
        "background": "Ambassador Williams has served in diplomatic roles for over 20 years, specializing in conflict resolution and international negotiations. They believe in the power of dialogue and have successfully mediated several international disputes.",
        "goals": ["Build consensus", "Maintain peace", "Foster understanding between parties"]
    },
    # --- Hospital crisis & emergency (Triage Stress-Tester) ---
    "Nurse Elena Vokshi": {
        "role": "doctor",
        "personality": {
            "openness": 0.6,
            "conscientiousness": 0.95,
            "extraversion": 0.7,
            "agreeableness": 0.7,
            "neuroticism": 0.5
        },
        "background": "Senior ED triage nurse with 18 years in mass-casualty response. She uses ESI/Manchester triage protocols under pressure, shields the waiting room from chaos, and fights anyone who skips the line or hoards resources. She has led triage during bus crashes and pandemic surges at a regional hospital.",
        "goals": [
            "Prioritize patients by medical urgency, not politics or loud voices",
            "Keep triage flow moving when arrivals spike",
            "Flag communication breakdowns before patients die in the hallway"
        ]
    },
    "Dr. Arben Krasniqi": {
        "role": "doctor",
        "personality": {
            "openness": 0.5,
            "conscientiousness": 0.9,
            "extraversion": 0.8,
            "agreeableness": 0.5,
            "neuroticism": 0.4
        },
        "background": "Director of resuscitation and emergency medicine. He runs the code team, assigns bays, and decides who gets immediate life support versus who waits. Calm in public, blunt in staff disputes. He measures success in minutes-to-treatment and hates when OR or ICU blocks his patients.",
        "goals": [
            "Save salvageable lives in the golden hour",
            "Secure resuscitation bays, ventilators, and crash carts",
            "Force clear handoffs between triage, trauma, and ICU"
        ]
    },
    "Dr. Luljeta Berisha": {
        "role": "doctor",
        "personality": {
            "openness": 0.7,
            "conscientiousness": 0.85,
            "extraversion": 0.6,
            "agreeableness": 0.4,
            "neuroticism": 0.35
        },
        "background": "Chief trauma surgeon on call during the surge. She competes for operating rooms, blood products, and anesthesia staff. She advocates for surgical candidates but accepts that not every red patient can go to OR immediately. Known for sharp conflicts with resuscitation over 'who is truly operable'.",
        "goals": [
            "Lock OR time for patients who benefit from surgery now",
            "Prevent futile operations when beds and blood are scarce",
            "Expose logistic bottlenecks (sterile supplies, transport, ICU beds)"
        ]
    },
    "Coordinator Fisnik Hoxha": {
        "role": "engineer",
        "personality": {
            "openness": 0.5,
            "conscientiousness": 0.9,
            "extraversion": 0.5,
            "agreeableness": 0.6,
            "neuroticism": 0.45
        },
        "background": "Hospital emergency logistics and blood-bank liaison. Tracks bed census, O-negative units, on-call staff, and ambulance offload delays. Speaks in numbers and timelines. Pushes back when clinicians demand resources that do not exist on the shelf.",
        "goals": [
            "Match limited beds, blood, and staff to realistic demand",
            "Surface supply chain failures before the ED gridlocks",
            "Coordinate transfers to sister hospitals when capacity is exceeded"
        ]
    },
    # --- Public policy & government response sandbox ---
    "Ardian Mulaj": {
        "role": "engineer",
        "personality": {
            "openness": 0.4,
            "conscientiousness": 0.8,
            "extraversion": 0.6,
            "agreeableness": 0.3,
            "neuroticism": 0.5
        },
        "background": "Skeptical business owner (retail and small construction). He worries about new taxes, permit delays, and vague 'reform' language. He reads draft policies for hidden costs and will threaten slowdowns, layoffs, or public campaigns if he feels ignored.",
        "goals": [
            "Protect business cash flow and predictable rules",
            "Expose vague or costly policy wording before adoption",
            "Force government to quantify economic impact in plain numbers"
        ]
    },
    "Valbona Sahiti": {
        "role": "rebel",
        "personality": {
            "openness": 0.85,
            "conscientiousness": 0.6,
            "extraversion": 0.85,
            "agreeableness": 0.35,
            "neuroticism": 0.55
        },
        "background": "Environmental and community activist focused on air quality, rivers, and public space. She organizes protests, uses social media aggressively, and distrusts developers and weak environmental impact studies. She frames every policy as a justice issue for ordinary families.",
        "goals": [
            "Block or rewrite policies that harm health and green space",
            "Name which phrases sound green but enable exploitation",
            "Mobilize public pressure when consultations feel fake"
        ]
    },
    "Deputy Ramadan Gashi": {
        "role": "diplomat",
        "personality": {
            "openness": 0.6,
            "conscientiousness": 0.75,
            "extraversion": 0.85,
            "agreeableness": 0.25,
            "neuroticism": 0.4
        },
        "background": "Opposition deputy in the municipal assembly. He attacks weak messaging, procedural shortcuts, and anything that smells like corruption. He is not always against change — he is against how change is sold. He will filibuster, leak drafts, and demand votes be delayed.",
        "goals": [
            "Find political vulnerabilities in the announcement strategy",
            "Force transparency and public debate before implementation",
            "Trigger administrative deadlock if process rules are broken"
        ]
    },
    "Citizen Enver Kelmendi": {
        "role": "teacher",
        "personality": {
            "openness": 0.6,
            "conscientiousness": 0.7,
            "extraversion": 0.5,
            "agreeableness": 0.75,
            "neuroticism": 0.6
        },
        "background": "Ordinary resident — works two jobs, uses public transport, has kids in local schools. He does not read legal PDFs. He reacts to tone, price tags, and trust. If policy language confuses or scares him, he calls the hotline, complains at the municipality, or simply disengages.",
        "goals": [
            "Understand what the policy means for his rent, bills, and commute",
            "Call out jargon and patronizing government speak",
            "Signal when messaging will backfire with normal people"
        ]
    },
    # --- Mock jury sandbox (12 juror profiles) ---
    "Juror 01 — Adem Hoxha": {
        "role": "engineer",
        "personality": {"openness": 0.4, "conscientiousness": 0.8, "extraversion": 0.4, "agreeableness": 0.5, "neuroticism": 0.5},
        "background": "Retired factory worker, age 58. Distrusts wealthy defendants and polished lawyers. Responds to plain talk about consequences. Mock jury panelist — react to defense arguments with gut sympathy or suspicion, cite exact phrases that moved you.",
        "goals": ["Punish arrogance and evasion", "Reward honest remorse if credible", "Flag lawyer jargon as insulting"]
    },
    "Juror 02 — Besa Krasniqi": {
        "role": "doctor",
        "personality": {"openness": 0.6, "conscientiousness": 0.85, "extraversion": 0.5, "agreeableness": 0.85, "neuroticism": 0.45},
        "background": "ER nurse, age 34. Strong empathy for victims and families. Sensitive to minimizing harm. Mock juror — note which defense phrases feel compassionate vs cold.",
        "goals": ["Protect vulnerable victims", "Reject excuses that blame the injured", "Trust medical facts over spin"]
    },
    "Juror 03 — Clarissa Moore": {
        "role": "engineer",
        "personality": {"openness": 0.5, "conscientiousness": 0.9, "extraversion": 0.6, "agreeableness": 0.35, "neuroticism": 0.4},
        "background": "Small business owner, age 45. Skeptical of excuses and delays. Values personal responsibility. Mock juror — flags vague timelines and 'mistakes were made' language.",
        "goals": ["Hold people accountable for choices", "Doubt sob stories without evidence", "Respect clear documentation"]
    },
    "Juror 04 — Daniel Kim": {
        "role": "scientist",
        "personality": {"openness": 0.75, "conscientiousness": 0.85, "extraversion": 0.4, "agreeableness": 0.55, "neuroticism": 0.35},
        "background": "Software engineer, age 29. Wants timelines, logs, and proof. Hates emotional manipulation without data. Mock juror — scores arguments on consistency and factual gaps.",
        "goals": ["Demand evidence chains", "Reject contradictions", "Separate emotion from provable facts"]
    },
    "Juror 05 — Elira Gashi": {
        "role": "teacher",
        "personality": {"openness": 0.65, "conscientiousness": 0.7, "extraversion": 0.55, "agreeableness": 0.8, "neuroticism": 0.55},
        "background": "Single mother of two, works in retail, age 38. Reads sincerity in tone. Tired of elites talking down. Mock juror — reacts to whether the defendant sounds like a real person or a script.",
        "goals": ["Spot patronizing lawyer speech", "Sympathize with relatable struggle if genuine", "Reject luxury-defendant detachment"]
    },
    "Juror 06 — Fatos Berisha": {
        "role": "diplomat",
        "personality": {"openness": 0.45, "conscientiousness": 0.9, "extraversion": 0.5, "agreeableness": 0.6, "neuroticism": 0.3},
        "background": "Army veteran, age 52. Respects duty, chain of command, and following rules. Mock juror — weighs whether the defendant honored obligations or shirked them.",
        "goals": ["Reward duty and discipline narratives", "Punish chaos and disregard for rules", "Distrust anti-institution rhetoric"]
    },
    "Juror 07 — Gentiana Rexhepi": {
        "role": "rebel",
        "personality": {"openness": 0.9, "conscientiousness": 0.45, "extraversion": 0.75, "agreeableness": 0.4, "neuroticism": 0.6},
        "background": "University student activist, age 22. Suspicious of police and corporate power. Mock juror — probes whether the state or corporation overreached.",
        "goals": ["Challenge prosecution framing", "Side with underdog if power abused", "Reject 'trust the system' without proof"]
    },
    "Juror 08 — Haki Murati": {
        "role": "teacher",
        "personality": {"openness": 0.5, "conscientiousness": 0.75, "extraversion": 0.45, "agreeableness": 0.65, "neuroticism": 0.4},
        "background": "Farmer, age 60. Practical, hates legal jargon. Asks 'what actually happened?' Mock juror — converts arguments into simple cause-and-effect.",
        "goals": ["Demand plain-language truth", "Reject clever word games", "Trust neighbors over experts if stories conflict"]
    },
    "Juror 09 — Ilir Panders": {
        "role": "spy",
        "personality": {"openness": 0.4, "conscientiousness": 0.8, "extraversion": 0.55, "agreeableness": 0.3, "neuroticism": 0.45},
        "background": "Former police officer, age 48. Pro-procedure, pro-victim, skeptical of defense theatrics. Mock juror — catches inconsistencies in alibis and body language cues described in testimony.",
        "goals": ["Catch holes in defense stories", "Support law enforcement when credible", "Punish obvious lies"]
    },
    "Juror 10 — Jonida Selimi": {
        "role": "teacher",
        "personality": {"openness": 0.8, "conscientiousness": 0.65, "extraversion": 0.6, "agreeableness": 0.9, "neuroticism": 0.5},
        "background": "Social worker, age 41. Believes in rehabilitation and context (trauma, poverty). Mock juror — notes which phrases open mercy vs which sound manipulative.",
        "goals": ["Consider life circumstances fairly", "Reject cruel dehumanizing language", "Spot performative remorse"]
    },
    "Juror 11 — Kujtim Bytyqi": {
        "role": "engineer",
        "personality": {"openness": 0.55, "conscientiousness": 0.95, "extraversion": 0.35, "agreeableness": 0.4, "neuroticism": 0.35},
        "background": "Accountant, age 50. Obsessed with numbers matching. Mock juror — attacks financial alibis, invoice gaps, and timeline math in the defense case.",
        "goals": ["Reconcile every number", "Flag spreadsheet contradictions", "Distrust round numbers and hand-waving"]
    },
    "Juror 12 — Linda Vokshi": {
        "role": "journalist",
        "personality": {"openness": 0.6, "conscientiousness": 0.9, "extraversion": 0.55, "agreeableness": 0.45, "neuroticism": 0.4},
        "background": "Retired court clerk, age 63. Knows procedure cold. Mock juror — punishes lawyers who skirt rules, mischaracterize evidence, or rush the jury.",
        "goals": ["Enforce fair process", "Reject improper argument tactics", "Document which phrases were objectionable"]
    },
    # --- Natural disaster relief logistics ---
    "Col. Driton Krasniqi": {
        "role": "engineer",
        "personality": {"openness": 0.45, "conscientiousness": 0.95, "extraversion": 0.7, "agreeableness": 0.4, "neuroticism": 0.25},
        "background": "Local military disaster-response commander. Controls airspace requests, heavy lift, convoy security, and bridge assessments. Insists on unified command. Clashes with NGOs that 'fly in and freelance'.",
        "goals": ["Secure helicopters and routes for life-saving ops", "Prevent chaotic civilian air traffic", "Enforce chain of command in the first 48 hours"]
    },
    "Mayor Shpresa Ahmeti": {
        "role": "diplomat",
        "personality": {"openness": 0.65, "conscientiousness": 0.75, "extraversion": 0.85, "agreeableness": 0.7, "neuroticism": 0.5},
        "background": "Municipal mayor of a flood-hit city. Accountable to citizens on live TV. Needs food, shelter names, and visible action. Fights anything that makes her commune look abandoned or bypassed.",
        "goals": ["Get visible relief to neighborhoods now", "Control narrative and local distribution points", "Avoid being overruled by army or foreign NGOs"]
    },
    "NGO Lead Sophie Laurent": {
        "role": "diplomat",
        "personality": {"openness": 0.8, "conscientiousness": 0.7, "extraversion": 0.75, "agreeableness": 0.65, "neuroticism": 0.45},
        "background": "Director of an international relief NGO just landed with cargo planes, volunteers, and donor reporting deadlines. Pushes fast deployment even if customs and military paperwork lag.",
        "goals": ["Move supplies from airport to survivors within 24h", "Protect donor credibility and media optics", "Resist bureaucratic freezes that waste perishable aid"]
    },
    "Red Cross Chief Naim Berisha": {
        "role": "doctor",
        "personality": {"openness": 0.55, "conscientiousness": 0.9, "extraversion": 0.6, "agreeableness": 0.85, "neuroticism": 0.4},
        "background": "National Red Cross logistics chief. Runs warehouses, volunteer lists, and family reunification. Neutral broker but angry when duplicate registration systems appear.",
        "goals": ["Single tracking system for aid and missing persons", "Fair distribution by need not politics", "Surface where food sits rotting in warehouses"]
    },
    "EMA Director Arta Bajrami": {
        "role": "engineer",
        "personality": {"openness": 0.6, "conscientiousness": 0.9, "extraversion": 0.65, "agreeableness": 0.55, "neuroticism": 0.35},
        "background": "Head of the national Emergency Management Agency. Owns the disaster declaration, inter-agency sitrep, and 48-hour priorities matrix. Maps 'death by bureaucracy' bottlenecks: customs, permits, overlapping commands.",
        "goals": ["Publish one incident command structure", "Cut duplicate needs assessments", "Name the exact hour where aid stalled and why"]
    },
}

# Scenario definitions
SCENARIOS = {
    "Political Negotiation": {
        "description": "A tense political negotiation where different parties must reach a compromise on a controversial issue.",
        "context": "A bill regarding environmental regulations vs. economic growth is being debated. Each party has different interests and constituencies to represent.",
        "phases": ["Opening statements", "Issue identification", "Negotiation", "Compromise seeking", "Final agreement"],
        "suggested_agents": ["Ambassador Williams", "Maya Rodriguez", "Dr. Sarah Chen"],
        "initial_prompt": "We need to discuss the proposed environmental regulations. The stakes are high for both the environment and the economy."
    },
    
    "Team Building": {
        "description": "A team of professionals working together to solve a complex problem.",
        "context": "A diverse team has been assembled to tackle a crisis situation that requires different expertise and perspectives.",
        "phases": ["Introductions", "Problem analysis", "Solution brainstorming", "Implementation planning", "Role assignment"],
        "suggested_agents": ["Dr. Sarah Chen", "Marcus Steel", "Agent X"],
        "initial_prompt": "We've been brought together to handle this crisis. Let's start by understanding what each of us brings to the table."
    },
    
    "Crisis Management": {
        "description": "A high-stakes crisis requiring immediate coordination and decision-making.",
        "context": "A natural disaster has struck, and the team must coordinate rescue efforts, resource allocation, and public communication.",
        "phases": ["Situation assessment", "Resource inventory", "Priority setting", "Action planning", "Execution coordination"],
        "suggested_agents": ["Dr. Sarah Chen", "Marcus Steel", "Ambassador Williams"],
        "initial_prompt": "The situation is critical. We need to act fast and coordinate our efforts. What's our immediate priority?"
    },
    
    "Corporate Espionage": {
        "description": "A covert operation where trust is scarce and everyone has hidden agendas.",
        "context": "Agents from different organizations are forced to work together while protecting their own interests and secrets.",
        "phases": ["Initial meeting", "Information sharing", "Trust building", "Revelation of motives", "Final confrontation"],
        "suggested_agents": ["Agent X", "Maya Rodriguez", "Marcus Steel"],
        "initial_prompt": "We all know why we're here, even if we can't say it openly. How do we proceed when trust is a luxury we can't afford?"
    },
    
    "Medical Ethics Committee": {
        "description": "A medical ethics committee debating a difficult case with moral implications.",
        "context": "A controversial medical case requires the committee to balance patient rights, medical ethics, resource allocation, and legal considerations.",
        "phases": ["Case presentation", "Ethical analysis", "Stakeholder perspectives", "Moral reasoning", "Decision making"],
        "suggested_agents": ["Dr. Sarah Chen", "Ambassador Williams", "Maya Rodriguez"],
        "initial_prompt": "We're here to discuss a case that challenges our understanding of medical ethics. Each perspective is important for reaching the right decision."
    },
    
    "Innovation Workshop": {
        "description": "A brainstorming session to develop innovative solutions to contemporary challenges.",
        "context": "A diverse group of experts is exploring cutting-edge solutions to global problems, balancing innovation with practical constraints.",
        "phases": ["Problem definition", "Creative exploration", "Feasibility analysis", "Prototype planning", "Implementation strategy"],
        "suggested_agents": ["Marcus Steel", "Dr. Sarah Chen", "Maya Rodriguez"],
        "initial_prompt": "Today we're thinking outside the box to solve problems that matter. What innovative approaches can we explore?"
    },

    "Hospital Crisis & Emergency (Triage Stress-Tester)": {
        "description": (
            "Simulim i krizave dhe emergjencave spitalore — agjentët luajnë role kritike "
            "(triazh, reanimacion, kirurgji, logjistikë) gjatë një incidenti masiv ose pandemie."
        ),
        "context": (
            "A regional hospital declares an internal mass-casualty incident: multiple trauma arrivals, "
            "ED at 140% capacity, only 2 ICU beds free, O-negative blood critically low, and 3 OR teams on call. "
            "Leadership runs a live stress-test war room to expose communication failures and resource conflicts "
            "before real patients pay the price. NOT real medical advice — training simulation only."
        ),
        "phases": [
            "Mass casualty alert",
            "Triage surge assessment",
            "Resource allocation conflict",
            "OR and ICU prioritization",
            "Blood bank and staffing crisis",
            "Debrief and communication gaps"
        ],
        "suggested_agents": [
            "Nurse Elena Vokshi",
            "Dr. Arben Krasniqi",
            "Dr. Luljeta Berisha",
            "Coordinator Fisnik Hoxha"
        ],
        "initial_prompt": (
            "Mass-casualty alert is active. Triage is overflowing, blood bank reports critical shortages, "
            "and three departments are demanding the same ICU bed. State your immediate priority and what "
            "resource you need in the next ten minutes."
        ),
        "target_users": "Hospital directors, emergency managers, department heads (ED, trauma, ICU, blood bank).",
        "value_proposition": (
            "Surfaces weak handoffs, hoarding, and logistic bottlenecks under scarcity — beds, blood, staff — "
            "in a safe sandbox before a real disaster."
        ),
        "default_win_goals": [
            "Triage protocol agreed under surge load",
            "ICU bed allocation decision documented",
            "Blood unit priority list established",
            "Clear handoff between resuscitation and surgery",
            "At least one communication gap named in debrief"
        ],
        "research_topic_hint": "hospital mass casualty triage resource allocation emergency",
    },

    "Public Policy & Government Response Sandbox": {
        "description": (
            "Sandbox për testimin e politikave publike dhe reagimit qeveritar — palë interesi "
            "simulojnë reagime para se një vendim të diskutueshëm të publikohet zyrtarisht."
        ),
        "context": (
            "A municipality (e.g. Prishtina or Mitrovica) or ministry prepares to launch a controversial measure: "
            "strict zoning reform, emergency fiscal rules, or a high-visibility infrastructure project. "
            "Before the press conference, communications staff run the draft announcement through a multi-agent "
            "sandbox. Agents play a skeptical businessman, environmental activist, opposition deputy, and an "
            "ordinary citizen. Goal: see which phrases trigger backlash, protests, or administrative deadlock."
        ),
        "phases": [
            "Policy draft presentation",
            "Stakeholder first reactions",
            "Media and public backlash test",
            "Administrative deadlock risk",
            "Communication rewrite",
            "Final announcement stress test"
        ],
        "suggested_agents": [
            "Ardian Mulaj",
            "Valbona Sahiti",
            "Deputy Ramadan Gashi",
            "Citizen Enver Kelmendi"
        ],
        "initial_prompt": (
            "Here is the draft policy headline and summary: a contested urban redevelopment with new environmental "
            "fees and fast-track permits. Each stakeholder — react in character. What words in this draft alarm you "
            "most, and what would make you block or protest the process?"
        ),
        "target_users": "Ministries, municipalities, public relations and communications offices.",
        "value_proposition": (
            "Shows which messaging triggers the harshest reactions or procedural blocks so teams can rewrite "
            "strategy before a public crisis."
        ),
        "default_win_goals": [
            "Each stakeholder's top objection captured",
            "At least one inflammatory phrase identified",
            "Plain-language rewrite suggested for citizens",
            "Deadlock risk (votes, permits, protests) assessed",
            "Revised talking points drafted for leadership"
        ],
        "research_topic_hint": "municipal policy public consultation backlash communication",
    },

    "Mock Jury Sandbox (Legal Strategy Tester)": {
        "description": (
            "Testuesi i strategjive gjyqësore — 12 agjentë jurorë me profile demografike dhe "
            "psikologjike të ndryshme reagojnë ndaj argumenteve mbrojtëse/prokuroriale."
        ),
        "context": (
            "A criminal trial mock deliberation room. Defense counsel has pasted a closing argument "
            "into the simulation (via moderator or initial prompt). Twelve jurors with distinct ages, "
            "jobs, and biases discuss which phrases create sympathy, doubt, or antipathy. "
            "NOT legal advice — litigation strategy training sandbox only. Law firms, prosecutors, "
            "and justice ministries use this instead of expensive human mock juries."
        ),
        "phases": [
            "Defense argument presented",
            "Jury first impressions",
            "Prosecution rebuttal stress test",
            "Deliberation — sympathy signals",
            "Deliberation — doubt and antipathy",
            "Panel lean report"
        ],
        "suggested_agents": [
            "Juror 01 — Adem Hoxha",
            "Juror 02 — Besa Krasniqi",
            "Juror 03 — Clarissa Moore",
            "Juror 04 — Daniel Kim",
            "Juror 05 — Elira Gashi",
            "Juror 06 — Fatos Berisha",
            "Juror 07 — Gentiana Rexhepi",
            "Juror 08 — Haki Murati",
            "Juror 09 — Ilir Panders",
            "Juror 10 — Jonida Selimi",
            "Juror 11 — Kujtim Bytyqi",
            "Juror 12 — Linda Vokshi",
        ],
        "initial_prompt": (
            "FICTIONAL CASE BRIEF (State v. Marin K., aggravated assault):\n"
            "Facts alleged: defendant struck the victim once during a parking dispute after victim "
            "blocked defendant's car; victim suffered a fractured cheekbone. Defense claims panic, "
            "no prior violence, immediate 911 call, full cooperation, and five years of voluntary "
            "payments to the victim's family for medical bills.\n\n"
            "DEFENSE CLOSING EXCERPT (react to THIS text, phrase by phrase):\n"
            "'My client made a terrible mistake in a moment of panic, but he returned to the scene, "
            "cooperated fully, and has devoted years to supporting the victim's family. The prosecution "
            "wants you to ignore remorse and context.'\n\n"
            "Each juror: independent reaction only. Quote defense words, label SYMPATHY/DOUBT/ANTIPATHY, "
            "state guilty / not guilty / undecided. Do not debate other jurors."
        ),
        "target_users": "Large law firms, prosecutors, Ministry of Justice, litigation strategy teams.",
        "value_proposition": (
            "Shows which argument phrases win sympathy vs trigger skepticism across a diverse jury panel — "
            "without paying for human mock juries."
        ),
        "default_win_goals": [
            "Each juror voiced at least one reaction to the defense",
            "Sympathy-triggering phrases identified",
            "Doubt/antipathy phrases identified",
            "Split or majority lean stated (guilty vs not guilty)",
            "At least one phrase rewrite suggested for counsel"
        ],
        "research_topic_hint": "",
        "agent_selection_note": "Select all 12 jurors. Use random or round-robin turn mode. Remote API recommended — 12 agents is slow on local GPU.",
        "prompt_mode": "mock_jury",
        "disable_auto_research": True,
        "director_note": (
            "MOCK JURY SANDBOX — litigation strategy training only, not legal advice.\n"
            "Each turn is ONE juror's independent worksheet row, not a group debate.\n"
            "Source text: defense closing + fictional case brief in the System message.\n"
            "Required labels per reply: PHRASE (quoted) | REACTION (SYMPATHY/DOUBT/ANTIPATHY) | "
            "REASON (one clause in juror's voice) | LEAN (guilty/not guilty/undecided).\n"
            "Forbidden: Yeah-but chains, copying prior juror metaphors, citing real celebrity trials, "
            "answering another juror's question, lawyer jargon."
        ),
    },

    "Natural Disaster Relief Logistics Coordinator": {
        "description": (
            "Koordinatori i logjistikës për fatkeqësi natyrore — tërmet ose përmbytje; ushtria, "
            "komuna, OJQ-të e huaja dhe Kryqi i Kuq përplasen për helikopterë, ushqim dhe komandë."
        ),
        "context": (
            "A 6.4 earthquake followed by flash flooding isolates three municipalities. First 48 hours: "
            "foreign NGO cargo lands at the airport, the army controls airspace, the mayor demands neighborhood "
            "distribution points, Red Cross runs family reunification, and EMA tries to publish unified command. "
            "Simulate where aid stalls — customs, duplicate assessments, helicopter turf wars, political photo-ops. "
            "Goal: map 'death by bureaucracy' before the next real disaster."
        ),
        "phases": [
            "First six hours — situation picture",
            "Foreign NGO arrival and customs",
            "Helicopter and airspace control fight",
            "Food and shelter distribution authority",
            "Chain of command clash",
            "48-hour bottleneck debrief"
        ],
        "suggested_agents": [
            "EMA Director Arta Bajrami",
            "Col. Driton Krasniqi",
            "Mayor Shpresa Ahmeti",
            "NGO Lead Sophie Laurent",
            "Red Cross Chief Naim Berisha",
        ],
        "initial_prompt": (
            "Earthquake-plus-flood: 200+ missing, two bridges down, airport receiving NGO flights but "
            "customs clearance averaging 11 hours. Who owns helicopters, who opens the first distribution "
            "hub, and what do you need in the next six hours? Speak as your agency — no vague promises."
        ),
        "target_users": "Military civil-affairs units, emergency management agencies, Red Cross / Red Crescent.",
        "value_proposition": (
            "Exposes turf wars and paperwork delays in the critical first 48 hours so agencies can write "
            "faster unified protocols before lives are lost to bureaucracy."
        ),
        "default_win_goals": [
            "Single incident command named (or failure documented)",
            "Helicopter priority list agreed or disputed clearly",
            "Food distribution point authority assigned",
            "Customs/warehouse bottleneck identified with timestamp",
            "48-hour debrief lists at least two bureaucratic death points"
        ],
        "research_topic_hint": "disaster relief logistics coordination earthquake flood humanitarian",
    },
}

class ScenarioManager:
    """Manages scenario selection, context, and progression"""
    
    def __init__(self):
        self.current_scenario = None
        self.current_phase = 0
        self.scenario_history = []
        self.custom_scenarios: Dict[str, Dict[str, Any]] = {}
    
    def get_available_scenarios(self) -> Dict[str, Dict]:
        """Get all available scenarios (built-in plus custom from UI)."""
        merged = {**SCENARIOS, **self.custom_scenarios}
        return merged

    def register_custom_scenario(self, name: str, data: Dict[str, Any]) -> None:
        """Register or overwrite a user-defined scenario (same shape as SCENARIOS values)."""
        self.custom_scenarios[name] = data

    def get_available_agents(self) -> Dict[str, Dict]:
        """Get all available agent configurations"""
        return AGENT_CONFIGS
    
    def clear_scenario(self) -> None:
        """Run free-form mode with no scripted scenario."""
        self.current_scenario = None
        self.current_phase = 0

    def set_scenario(self, scenario_name: str):
        """Set the current scenario"""
        all_scenarios = {**SCENARIOS, **self.custom_scenarios}
        if scenario_name in all_scenarios:
            self.current_scenario = all_scenarios[scenario_name]
            self.current_phase = 0
            self.scenario_history.append({
                "scenario": scenario_name,
                "start_time": datetime.now(),
                "phases_completed": []
            })
        else:
            raise ValueError(f"Scenario '{scenario_name}' not found")
    
    def get_current_context(self) -> Dict[str, Any]:
        """Get current scenario context"""
        if not self.current_scenario:
            return {}
        
        return {
            "scenario_description": self.current_scenario["description"],
            "scenario_context": self.current_scenario["context"],
            "current_phase": self.current_scenario["phases"][self.current_phase] if self.current_phase < len(self.current_scenario["phases"]) else "Conclusion",
            "phase_number": self.current_phase + 1,
            "total_phases": len(self.current_scenario["phases"]),
            "initial_prompt": self.current_scenario.get("initial_prompt", ""),
            "target_users": self.current_scenario.get("target_users", ""),
            "value_proposition": self.current_scenario.get("value_proposition", ""),
            "default_win_goals": self.current_scenario.get("default_win_goals", []),
            "research_topic_hint": self.current_scenario.get("research_topic_hint", ""),
            "agent_selection_note": self.current_scenario.get("agent_selection_note", ""),
            "prompt_mode": self.current_scenario.get("prompt_mode", ""),
            "disable_auto_research": self.current_scenario.get("disable_auto_research", False),
            "director_note": self.current_scenario.get("director_note", ""),
        }
    
    def advance_phase(self):
        """Move to the next phase of the scenario"""
        if self.current_scenario and self.current_phase < len(self.current_scenario["phases"]) - 1:
            self.current_phase += 1
            if self.scenario_history:
                self.scenario_history[-1]["phases_completed"].append(self.current_scenario["phases"][self.current_phase - 1])
    
    def get_suggested_agents(self, scenario_name: str) -> List[str]:
        """Get suggested agents for a scenario"""
        all_scenarios = {**SCENARIOS, **self.custom_scenarios}
        if scenario_name in all_scenarios:
            return all_scenarios[scenario_name].get("suggested_agents", [])
        return []
    
    def generate_random_scenario_context(self) -> str:
        """Generate additional random context for variety"""
        contexts = [
            "Tensions are high and time is running short.",
            "New information has just come to light that changes everything.",
            "An unexpected stakeholder has entered the conversation.",
            "Previous assumptions are being challenged.",
            "A deadline is approaching rapidly.",
            "External pressures are mounting.",
            "Public opinion is shifting.",
            "Technical constraints have been discovered.",
            "Budget limitations have been revealed.",
            "Legal implications are becoming clear."
        ]
        return random.choice(contexts)
    
    def get_phase_transition_prompt(self) -> str:
        """Get a prompt for transitioning between phases"""
        if not self.current_scenario:
            return ""
        
        current_phase_name = self.current_scenario["phases"][self.current_phase] if self.current_phase < len(self.current_scenario["phases"]) else "Conclusion"
        
        transitions = {
            "Opening statements": "Let's begin by having each party state their position clearly.",
            "Issue identification": "Now let's identify the key issues that need to be addressed.",
            "Problem analysis": "Let's analyze the core problems we're facing.",
            "Solution brainstorming": "It's time to brainstorm potential solutions.",
            "Negotiation": "Let's start the formal negotiation process.",
            "Compromise seeking": "We need to find middle ground that works for everyone.",
            "Implementation planning": "How do we put our ideas into action?",
            "Final agreement": "Let's work towards a final agreement.",
            "Conclusion": "Let's wrap up and summarize what we've accomplished.",
            # Hospital crisis & emergency
            "Mass casualty alert": "The mass-casualty code is live. Each lead states situational awareness and immediate needs.",
            "Triage surge assessment": "Triage reports patient categories and flow. Challenge any bypass of protocol.",
            "Resource allocation conflict": "Departments compete for beds, staff, and equipment — justify every claim with urgency.",
            "OR and ICU prioritization": "Surgery and ICU leads negotiate who gets the next slot and who must wait or transfer.",
            "Blood bank and staffing crisis": "Logistics reports shortages. Clinicians must reprioritize without hiding the trade-offs.",
            "Debrief and communication gaps": "Name what broke down in handoffs, language, or authority — no blame theater, just fixes.",
            # Public policy sandbox
            "Policy draft presentation": "Government side presents the draft policy in the words they plan to use publicly.",
            "Stakeholder first reactions": "Each interest group reacts to the draft — focus on trigger words and trust.",
            "Media and public backlash test": "Simulate headlines, social posts, and citizen hotline complaints.",
            "Administrative deadlock risk": "Identify votes, permits, lawsuits, or protests that could freeze implementation.",
            "Communication rewrite": "Propose concrete rewrites — shorter sentences, plain language, honest trade-offs.",
            "Final announcement stress test": "Read the revised announcement aloud. Stakeholders give a final pass or fail.",
            # Mock jury sandbox
            "Defense argument presented": "Counsel's argument is read aloud. Jurors listen — no deliberation yet, only notes.",
            "Jury first impressions": "Each juror gives a one-paragraph gut reaction — sympathy, doubt, or anger.",
            "Prosecution rebuttal stress test": "Prosecution counters the defense framing. Jurors note what flipped or hardened their view.",
            "Deliberation — sympathy signals": "Jurors cite exact phrases that made them lean toward mercy or acquittal.",
            "Deliberation — doubt and antipathy": "Jurors cite phrases that sounded false, manipulative, or guilty.",
            "Panel lean report": "Foreperson-style summary: vote lean count, hung issues, and top three phrases to rewrite.",
            # Disaster relief logistics
            "First six hours — situation picture": "Each agency reports damage, assets on hand, and the single biggest blocker.",
            "Foreign NGO arrival and customs": "NGO cargo is on the tarmac — who clears it and how fast?",
            "Helicopter and airspace control fight": "Military, mayor, and NGO compete for rotors and landing zones.",
            "Food and shelter distribution authority": "Who runs the hubs, who signs off, who gets photo credit?",
            "Chain of command clash": "EMA pushes unified command — who refuses to subordinate?",
            "48-hour bottleneck debrief": "Name the hour aid died in a warehouse, queue, or meeting.",
        }
        
        return transitions.get(current_phase_name, f"We're now in the {current_phase_name} phase.")
    
    def get_scenario_presets(self, scenario_name: str) -> Dict[str, Any]:
        """Win goals and research hints bundled with a scenario."""
        all_scenarios = {**SCENARIOS, **self.custom_scenarios}
        if scenario_name not in all_scenarios:
            return {}
        data = all_scenarios[scenario_name]
        return {
            "default_win_goals": data.get("default_win_goals", []),
            "research_topic_hint": data.get("research_topic_hint", ""),
            "initial_prompt": data.get("initial_prompt", ""),
            "director_note": data.get("director_note", ""),
            "prompt_mode": data.get("prompt_mode", ""),
            "disable_auto_research": data.get("disable_auto_research", False),
        }

    def export_scenario_history(self) -> str:
        """Export scenario history as JSON"""
        return json.dumps(self.scenario_history, default=str, indent=2)
