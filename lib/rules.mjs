export const lexicon = {
  urgency: ["now", "hurry", "last chance", "right away"],
  guilt: ["for grandma", "if you cared", "real friends would"],
  authority: ["experts say", "everyone knows", "studies prove"],
  vagueness: ["soon", "later", "eventually", "somehow"],
  agency_theft: ["just do it", "don't think", "trust me"],
  commitment: ["we will", "i will", "we should", "let's", "we need to"]
};

export const rules = [
  {
    id: "urgency_without_deadline",
    when: { any: lexicon.urgency, not: { hasDeadline: true } },
    outcome: { frame: "urgency", needs: ["deadline"], severity: 2 }
  },
  {
    id: "guilt_hook",
    when: { any: lexicon.guilt },
    outcome: { frame: "guilt", needs: ["options"], severity: 3, agencyDelta: -2 }
  },
  {
    id: "authority_without_evidence",
    when: { any: lexicon.authority, not: { hasEvidence: true } },
    outcome: { frame: "authority", needs: ["evidence"], severity: 2 }
  },
  {
    id: "vagueness_without_deadline",
    when: { any: lexicon.vagueness, not: { hasDeadline: true } },
    outcome: { frame: "vagueness", needs: ["deadline"], severity: 1 }
  },
  {
    id: "commitment_ambiguity_missing_owner",
    when: { any: ["we will", "we should", "we need to"], not: { missingSlots: ["owner"] } },
    outcome: { frame: "commitment_ambiguity", needs: ["owner"], severity: 2 }
  },
  {
    id: "agency_theft_no_options",
    when: { any: lexicon.agency_theft },
    outcome: { frame: "agency_theft", needs: ["options"], severity: 4, agencyDelta: -3 }
  },
  {
    id: "other_attribution_risk",
    when: { obs: "other", any: ["they meant", "you wanted", "he knew", "she knew"] },
    outcome: { frame: "attribution", needs: ["evidence"], severity: 2 }
  }
];
