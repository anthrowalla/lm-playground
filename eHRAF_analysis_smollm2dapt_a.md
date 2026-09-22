# eHRAF OCM paragraph tagging — smollm2_dapt_tag review sample

Paragraph-level OCM subject-code predictions from `smollm2_dapt_tag`
(362M parameters: SmolLM2-360M base further domain-adaptively pretrained on the same eHRAF-derived v5 corpus mixed with general-web replay, then fine-tuned for OCM tagging). Decoding is greedy (deterministic); predictions are
filtered to the 743 codes in the published OCM code list.

Each paragraph is presented with a context header: the section title path
plus the OCM codes assigned to the *other* paragraphs of the same section
(leave-one-out). In this sample those sibling codes are the original
analysts' assignments; in the intended assist flow they would come from the
analyst's own earlier paragraphs instead.

We are asking HRAF analysts to review this sample and note, per paragraph:
where the coding deviates significantly from the original, where it is
downright wrong, and their impression of what is right — and, where they
can, *why* they think the model went wrong or right in its choices.

## sk15 — Lengua festivals and functional substitutes

**SECTION:** Lengua Festivals and Functional Substitutes / III / Major functions of the shaman  (6 paragraphs) — hdocid sk15-004

### Paragraph 1

One of the duties of the shaman was the acquiring of spirit consent for the group to camp in a given area and to exploit the food resources extant there. Since each area was viewed as having a spirit owner, such spirit owners had to be consulted and their permission was a prerequisite to any exploitation of the resources. It was the duty of the shaman during the first night that the group camped at a given place to make this spirit contact and to get the permission or receive the denial of privileges.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 221 | ANNUAL CYCLE |  | x |
| 361 | SETTLEMENT PATTERNS |  | x |
| 756 | SHAMANS AND PSYCHOTHERAPISTS | x | x |
| 776 | SPIRITS AND GODS | x | x |
| 787 | REVELATION AND DIVINATION | x |  |
| 823 | ETHNOGEOGRAPHY |  | x |

Hits: 2 of 5 actual codes predicted · False positives: 1.

### Paragraph 2

Furthermore, it was the shaman's responsibility to be alert to any lurking harmful spirits. Such spirits came either of their own accord or they were sent by enemy shamans. When any such spirits threatened the group it was the shaman's task to sing and “disarm” the spirits, for shamanistic power songs always “tamed” and “charmed” fierce evil spirits. Since, however, they did not want to be “charmed” to the point of being identified, the spirits generally fled as soon as they had lost their “ferocity.” Under certain circumstances it was necessary also to purify the people and the environment. Thus once the shaman had ascertained that evil spirits were indeed in the environment, he could arrange for the fumigation of the village either through the burning of palo santo wood (an oil-bearing wood which gives off a very pungent odor when burned) and letting smoke pass over the entire village, or by having all the individuals wash themselves in a palo santo solution. This removed from the individual the susceptibility to the attacking quilyicjama and served as a type of immunization.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 753 | THEORY OF DISEASE | x |  |
| 755 | MAGICAL AND MENTAL THERAPY | x |  |
| 756 | SHAMANS AND PSYCHOTHERAPISTS | x | x |
| 776 | SPIRITS AND GODS | x | x |
| 783 | PURIFICATION AND ATONEMENT | x | x |
| 788 | RITUAL |  | x |
| 789 | MAGIC | x |  |

Hits: 3 of 4 actual codes predicted · False positives: 3.

### Paragraph 3

In case some malignant spirit had already brought illness to a member of the group, it was the shaman's responsibility to help restore the individual to health. In cases of serious illness very frequently shamans from several groups banded together around the sick individual to effect the cure. This participation in curing in other bands is interesting, for it not only served as an augmentation of spirit power, but it was also a public affirmation that their band was not the one guilty of sending the harmful spirits.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 628 | INTER-COMMUNITY RELATIONS | x | x |
| 753 | THEORY OF DISEASE | x | x |
| 755 | MAGICAL AND MENTAL THERAPY | x |  |
| 756 | SHAMANS AND PSYCHOTHERAPISTS | x | x |
| 776 | SPIRITS AND GODS |  | x |

Hits: 3 of 4 actual codes predicted · False positives: 1.

### Paragraph 4

Possibly the most dramatic function of the shaman came in the case of an epidemic or in the case of severe drought. These crisis experiences were the great tests of the shaman's power; for at such times the threat rested on the whole group. The shaman was expected to throw all his resources into play to, stem the tide of the epidemic or to end the drought by the production of rain.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 731 | DISASTERS | x | x |
| 756 | SHAMANS AND PSYCHOTHERAPISTS | x | x |
| 789 | MAGIC | x |  |
| 821 | ETHNOMETEOROLOGY |  | x |

Hits: 2 of 3 actual codes predicted · False positives: 1.

### Paragraph 5

The rain-making ceremony might take place in an individual band or could take place during the extensive festival season near the algarrobo forests where many bands had gathered. Since the algarrobo harvest corresponded with the dry season, the shamans from a number of bands participated. The shaman (or shamans) and the group of “laymen” danced and chanted until the shaman had partaken of enough liquor to “lighten” his soul. He then “sent” his soul to the “sea of the north” where the “souls of the birds of myth age” were viewed as lying. Present-day birds do not have souls. It was the duty of these bird souls to bring the rains. Each bird had several gourds which it filled with water and put under its wings. Then it flew two days to the south to deposit its rain in the Chaco. Occasionally these birds fell asleep. It was then the duty of the shaman to sing and to awaken them.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 221 | ANNUAL CYCLE | x |  |
| 273 | ALCOHOLIC BEVERAGES | x | x |
| 533 | MUSIC | x |  |
| 535 | DANCE | x | x |
| 755 | MAGICAL AND MENTAL THERAPY |  | x |
| 756 | SHAMANS AND PSYCHOTHERAPISTS | x | x |
| 774 | ANIMISM | x | x |
| 776 | SPIRITS AND GODS | x | x |
| 788 | RITUAL |  | x |
| 789 | MAGIC | x | x |
| 821 | ETHNOMETEOROLOGY | x | x |

Hits: 7 of 9 actual codes predicted · False positives: 2.

### Paragraph 6

Far more serious, however, was the condition when some enemy quilyicjama spirits had laid seige on the bird-souls to prevent them from bringing rain. In such cases the lightened shaman soul had to first identify the source of the trouble; then sing a different song which was to charm the enemy spirit forces so that they would become “tame.” Once robbed of their “ferocity” the spirits generally fled because they did not want to be lulled into being identified (and being caught).

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 533 | MUSIC |  | x |
| 755 | MAGICAL AND MENTAL THERAPY | x |  |
| 756 | SHAMANS AND PSYCHOTHERAPISTS | x | x |
| 774 | ANIMISM | x |  |
| 776 | SPIRITS AND GODS | x | x |
| 789 | MAGIC | x |  |
| 821 | ETHNOMETEOROLOGY |  | x |

Hits: 2 of 4 actual codes predicted · False positives: 3.

## ef05 — Montenegrin social organization and values: political ethnography of a refuge area tribal adaptation

**SECTION:** Chapter XI VALUES, RELIGION, AND MORALITY / Moral Sanctions  (4 paragraphs) — hdocid ef05-001

### Paragraph 1

To a certain extent, the Montenegrin moral system ran itself. People gossiped, and every individual knew that such talk and the resulting evaluation of moral reputations could blacken or whiten one’s ‘obraz’ depending upon what was known or suspected. Thus, gossip served as an indirect (and unintentional) mechanism of social sanctioning.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 577 | ETHICS | x |  |
| 626 | SOCIAL CONTROL | x | x |

Hits: 1 of 1 actual codes predicted · False positives: 1.

### Paragraph 2

Blood feud, as a very different kind of mechanism, functioned in a similarly unintentional way. Retaliation was not carried out for the conscious purpose of reinforcing moral values; indeed, many of the behaviors which began feuds were morally neutral or morally ambiguous. ^52 But Montenegrins were well aware of the behaviors most likely to provoke feuds. These included dishonoring a previously honorable maiden and refusing to marry her, carrying on with another man’s wife, insulting a man in a way which became unbearable, or killing another person for whatever reason, except where special circumstances exempted one from retaliation. Montenegrins were also well aware of the circumstances which called for vengeance killing on a virtually mandatory basis. Because of this predictability, the feud functioned as an indirect social sanction. A tribesman thought twice before he indulged himself in a behavior likely to provoke a feud, because he knew he would be the primary target, and that if he escaped someone else in his household or clan was likely to pay. ^52

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 000 | MATERIAL NOT RELEVANT |  | x |
| 577 | ETHICS | x | x |
| 627 | INFORMAL IN-GROUP JUSTICE | x | x |
| 681 | SANCTIONS |  | x |

Hits: 2 of 4 actual codes predicted · False positives: 0.

### Paragraph 3

Other behaviors which helped the tribal moral system to work were far more intentional. Informal sanctions involving deliberate application of social pressure have been mentioned, including ostracism, ridicule, and direct criticism. Furthermore, the entire tribe could gather to sentence a traitor to death or exile. Another conscious concern was with internal social harmony, since very deliberate efforts were made to pacify quarrels before these turned into feuds. Ongoing feuds, as we have seen, were also pacified once the feuding parties could be persuaded to set aside honor and accept money in lieu of taking blood. Efforts by those not involved were sophisticated, vigorous and concerned, although they had no actual power to interfere. Often priests, with the large respect they received, were key persons in such negotiations.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 577 | ETHICS | x | x |
| 578 | INGROUP ANTAGONISMS |  | x |
| 619 | TRIBE AND NATION | x |  |
| 626 | SOCIAL CONTROL | x |  |
| 627 | INFORMAL IN-GROUP JUSTICE | x | x |

Hits: 2 of 3 actual codes predicted · False positives: 2.

### Paragraph 4

This intentional side of social sanctioning was directly shaped by the moral values we have discussed. For example, political unity of the tribe and clan was especially valued because this related to continuance of political autonomy, while the emphasis placed on every male’s being a warrior stemmed from a similar concern. Thus Montenegrins values set up social goals which oriented the deliberate efforts of Montenegrins to control and shape their own social life. In the absence of any external regulation, they were responsible for their own moral order, and their values guided them in deliberately imposing negative or positive sanctions to change the behavior of individuals in desired directions.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 577 | ETHICS | x | x |
| 613 | LINEAGES |  | x |
| 619 | TRIBE AND NATION | x | x |
| 626 | SOCIAL CONTROL | x | x |

Hits: 3 of 4 actual codes predicted · False positives: 0.

## ef05 — Blood revenge: the anthropology of feuding in Montenegro and other tribal societies

**SECTION:** 9 Making Further Sense of the Feud / CLAN COMPETITION WITHIN THE POLITICAL SYSTEM  (3 paragraphs) — hdocid ef05-002

### Paragraph 1

We may now compare the situation of tribes that were competing with other tribes and the situation of clans that were competing within a tribe. Between neighboring tribes there was always the latent struggle for territory, which was kept in equilibrium by careful balancing of power but without any coercive mechanism to control conflicts. Among the clans there reigned also a generally competitive situation, but in the absence of territorial stakes, careful balancing of power was not necessary. Members of small and large clans were equally members of the tribal community, and external warfare provided an arena in which all of the clans could simultaneously cooperate and vie for heroic status. Thus, while the larger or more heroic clans were more influential in determining tribal policies, every clan had a right to exist peaceably within the tribal moral community.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 121 | THEORETICAL ORIENTATION | x |  |
| 614 | CLANS | x | x |
| 619 | TRIBE AND NATION | x | x |
| 628 | INTER-COMMUNITY RELATIONS | x | x |

Hits: 3 of 3 actual codes predicted · False positives: 1.

### Paragraph 2

This meant that there was little need for clans to form alliances with respect to warfare within the tribe, since the idea of such warfare went against everything the tribe stood for. But if conflicts did arise within the tribe, this absence of alliances left the two clans pretty much on their own in resolving the conflict. This put smaller clans at a marked disadvantage, because a man who belonged to a very small clan could be driven from the tribe if he killed or seriously insulted a member of a much larger clan. If the initial retaliation involved his entire clan as target, then the entire clan might have to emigrate. Such actions are poorly documented, because obviously no written pacifications took place. But many feuds took this direction; this is apparent from the very large number of refugee clans in Montenegro and from the large number of Montenegrins who had to settle elsewhere “because of blood.”

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 167 | EXTERNAL MIGRATION | x | x |
| 614 | CLANS | x | x |
| 628 | INTER-COMMUNITY RELATIONS | x | x |

Hits: 3 of 3 actual codes predicted · False positives: 0.

### Paragraph 3

In historical descriptions of longer-lasting feuds the issue of the relative size of clans is never raised, but this important aspect must not be overlooked. I believe that the feuding system was founded on the unstated assumption that whenever the two groups involved were sufficiently at parity so that use of coercive force could not resolve the issue without all but destroying both groups, then it was time for the complex of rules pertaining to the middle game to go into effect. Thus, it was necessary that some minimal degree of parity in clan strength and psychological motivation exist, before the alternating-homicide rules of middle-game feuding could be applied as a solution to the problem of internal conflict.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 121 | THEORETICAL ORIENTATION | x | x |
| 614 | CLANS | x | x |
| 627 | INFORMAL IN-GROUP JUSTICE | x | x |
| 628 | INTER-COMMUNITY RELATIONS | x | x |

Hits: 4 of 4 actual codes predicted · False positives: 0.

## st13 — Land, politics, and ethnicity in a Carib Indian community

**SECTION:** Land, Politics, and Ethnicity in a Carib Indian Community / THE EMERGENCE OF MODERN CARIB ETHNICITY / Reasons for Emergence  (4 paragraphs) — hdocid st13-015

### Paragraph 1

The Caribs were among the poorest, least educated people in the island. However, it was not long before they realized that within the context of lower class Dominican society, they possessed an advantage which other rural cultivators lacked: while other farmers had to buy land, they had free access to 3,700 acres of land. Since the widespread practice of swidden cultivation necessitated the planting of a new garden every few years, farmers had to have access to a considerable amount of land in order to meet their families' needs. The Caribs were well aware that their possession of a 3,700-acre reserve conferred on them a distinct advantage. As one woman pointed out: “Caribs are apart, for we have our own land. We can work where we feel like. Outsiders have to buy their land.”

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 186 | CULTURAL IDENTITY AND PRIDE | x |  |
| 241 | TILLAGE |  | x |
| 423 | REAL PROPERTY | x | x |
| 432 | BUYING AND SELLING | x |  |
| 657 | PUBLIC WELFARE |  | x |

Hits: 1 of 3 actual codes predicted · False positives: 2.

### Paragraph 2

But this advantage was one that they had to struggle to maintain. Although the land was officially organized into a reserve, the Caribs' right to occupy the land was never actually guaranteed by formal ordinance. Consequently, the Indians believed that they were at the mercy of the government not to dissolve the reserve. Present-day Caribs frequently tell the following story: ' During the early 1900s a government inspector visited the Carib chief and warned him not to allow strangers to mix up the race. Otherwise, the official explained, the government would say it had [there were] no Caribs again and the land donated to them would be taken away. ' They also feared the appropriation of their land by non-Carib men who came to live on the reserve. As one Carib woman explained: ' When outsiders come in they fight for your land. They want your plantation. The strangers bringing plenty trouble in the place. ' The Caribs realized that in order to keep their territory intact they would have to achieve two objectives: (1) convince the government to maintain the reserve for them, and (2) regulate access to reserve land by non-Caribs. The attainment of these goals was contingent on their ability to effect some sort of political organization. So the Caribs gradually articulated an informal political organization cemented by their ethnic identity.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 186 | CULTURAL IDENTITY AND PRIDE | x | x |
| 423 | REAL PROPERTY | x | x |
| 657 | PUBLIC WELFARE | x | x |

Hits: 3 of 3 actual codes predicted · False positives: 0.

### Paragraph 3

The political nature of ethnicity has been stressed recently by a number of social scientists. Sanders (1972), for example, shows how the Amerindians of Guyana have used their distinct identity as a basis for organizing formal political associations. According to Sanders, one goal espoused by these groups is the persuasion of the local government to give the Amerindian people ownership rights to their land. These Amerindians live in specially designated areas on land officially owned by the government. They fear that unless they can acquire title to the land, the government will eventually dispose of it.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 171 | COMPARATIVE EVIDENCE | x | x |
| 186 | CULTURAL IDENTITY AND PRIDE | x | x |
| 423 | REAL PROPERTY | x | x |

Hits: 3 of 3 actual codes predicted · False positives: 0.

### Paragraph 4

In a study of the Ute Indians of Utah, Collins (1973) shows how the desire to defend the Ute reservation and preserve their lifestyle underlies recent expressions of Ute ethnicity. Hicks and Kertzer (1972), on the other hand, point out how the highly acculturated “Monhegan” Indians of New England use their ethnicity to avoid being classified as black by members of the surrounding community. As the authors point out, it is more politically and economically advantageous to be considered Indian than it is to be labeled “black.” Paredes (1974), in a study of the Creek Indians of Alabama, shows how the desire to benefit from land claims money has prompted these people to stress their Indian identity. Paredes also points out that the Creeks have used ethnicity to organize a political party and to obtain certain privileges. In an earlier study, Cohen (1969) demonstrates how Hausa migrants in Yoruba towns have manipulated their ethnic identity to establish an informal political group whose major objectives are to protect and to coordinate a long-distance trading monopoly.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 171 | COMPARATIVE EVIDENCE | x | x |
| 186 | CULTURAL IDENTITY AND PRIDE | x | x |
| 423 | REAL PROPERTY |  | x |

Hits: 2 of 3 actual codes predicted · False positives: 0.

## st13 — The ethnobotany of the Island Caribs of Dominica

**SECTION:** section / IV. CATALOGUE OF THE KNOWN ECONOMIC PLANTS OF THE DOMINICA CARIBS / CHENOPODIACEAE. GOOSEFOOT FAMILY.  (6 paragraphs) — hdocid st13-013

### Paragraph 1

Chenopodium ambrosioides Linnaeus, Sp. Pl. 219. 1753.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 278 | PHARMACEUTICALS | x |  |
| 824 | ETHNOBOTANY | x | x |

Hits: 1 of 1 actual codes predicted · False positives: 1.

### Paragraph 2

Chenopodium anthelminticum Linnaeus, Sp. Pl. 220. 1753.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 278 | PHARMACEUTICALS | x |  |
| 824 | ETHNOBOTANY | x | x |

Hits: 1 of 1 actual codes predicted · False positives: 1.

### Paragraph 3

Wormseed. — Semen Contra.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 278 | PHARMACEUTICALS | x |  |
| 824 | ETHNOBOTANY | x | x |

Hits: 1 of 1 actual codes predicted · False positives: 1.

### Paragraph 4

Wormseed is a strong-scented annual plant widely distributed in the temperate and tropical regions of both the Old and New World.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 278 | PHARMACEUTICALS | x |  |
| 824 | ETHNOBOTANY | x | x |

Hits: 1 of 1 actual codes predicted · False positives: 1.

### Paragraph 5

A «tea» made from its leaves is drunk by the Carib as a beverage or is used to deworm children. The leaves are also one constituent of a ritual bath given to the mother after childbirth (see Ocimum micranthum ).

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 278 | PHARMACEUTICALS | x | x |
| 824 | ETHNOBOTANY | x | x |
| 846 | POSTNATAL CARE | x | x |

Hits: 3 of 3 actual codes predicted · False positives: 0.

### Paragraph 6

Specimen citation: 3364.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 278 | PHARMACEUTICALS | x |  |
| 824 | ETHNOBOTANY | x | x |

Hits: 1 of 1 actual codes predicted · False positives: 1.

## fc07 — Rainforest relations: gender and resource use among the Mende of Gola, Sierra Leone

**SECTION:** PART TWO Gender and resource use in the Gola forest of Sierra Leone / 8 MONEY, FOOD AND MANAGING / CONCLUSION  (4 paragraphs) — hdocid fc07-009

### Paragraph 1

People draw on a wide range of social relations in the acquisition, control and use of money and food resources. This makes it impossible to understand the local economy by focusing on particular social units, such as kitchens or ‘households’. In as much as these exist, important activities and relationships constantly cross-cut their boundaries (cf. Guyer and Peters, 1987; Leach, 1991b). Nor is it possible to understand women's use of money and food by isolating current female activities for analysis, as some studies of women's socio-economic and environmental roles do. Instead, people are embedded in nexuses of specific opportunities and obligations associated with kinship, friendship and patron-client relations, and their experiences depend on their ability to manage and draw on these effectively. This applies equally to different women and men, although there is considerable variation in their needs and in the claims and relations on which they are able to draw. In the context of the land-use changes on the northern edge of Gola North, I have drawn attention, for example, to the growing disjunctions between women's limited cash incomes and high expenditure needs, and to the increasing ambiguity surrounding men's provisioning responsibilities and contributions. Whereas older women derive support from their adult children and some wives are assisted by natal kin living nearby, other women cope by investing on their own accounts in subsistence activities and social networks, and by resorting to strategies such as covert resource appropriation in the ambiguous spaces between what is and is not publicly acceptable.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 121 | THEORETICAL ORIENTATION | x | x |
| 434 | INCOME AND DEMAND | x |  |
| 454 | SAVING AND INVESTMENT |  | x |
| 562 | GENDER STATUS | x | x |
| 571 | SOCIAL RELATIONSHIPS AND GROUPS | x | x |
| 593 | FAMILY RELATIONSHIPS |  | x |

Hits: 3 of 5 actual codes predicted · False positives: 1.

### Paragraph 2

These social networks and patron-client relations are central both to people's security and to their acquisition of wealth and status. The monetary economy is not leading to the disintegration of these ‘wealth in people’ relations; instead, money is increasingly an input to and medium for them. Money has, however, altered the opportunities for certain groups of people to acquire independence from others' control, and acquire clients for themselves. Financial acumen now rests alongside high-status forest-related activities as a means to attract and keep followers, and this has added to tensions in village life, whether between older and younger people or men and women.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 121 | THEORETICAL ORIENTATION | x | x |
| 434 | INCOME AND DEMAND | x |  |
| 556 | ACCUMULATION OF WEALTH | x |  |
| 562 | GENDER STATUS |  | x |
| 571 | SOCIAL RELATIONSHIPS AND GROUPS | x | x |

Hits: 2 of 3 actual codes predicted · False positives: 2.

### Paragraph 3

This chapter has shown how the resource-using activities presented in earlier chapters are currently integrated within monetary and consumptive relations. Annual-cropping, tree-cropping, and wild plant and animal use are all at least partly oriented towards acquiring money and food, and certain aspects of these become clearer when seen in terms of people's concerns over ‘managing’. Current socio-economic struggles shape such recent resource-use practices as hunting for the market, sales of local timber and non-timber forest products, and the cultivation of cassava, groundnuts and vegetables. Money and food-use also shape (and are shaped by) natural resource management relations, as when patrons make claims on their financially indebted clients' agricultural labour, or when sharing food at a family sacrifice to ancestors reinforces people's common membership of a family ( mbonda ) group and their acceptance of its head's authority, of importance in land and other resource-tenure arrangements.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 121 | THEORETICAL ORIENTATION | x | x |
| 178 | SOCIOCULTURAL TRENDS |  | x |
| 185 | CULTURAL GOALS | x |  |
| 311 | LAND USE | x | x |
| 314 | FOREST PRODUCTS |  | x |
| 434 | INCOME AND DEMAND | x |  |
| 571 | SOCIAL RELATIONSHIPS AND GROUPS | x |  |

Hits: 2 of 4 actual codes predicted · False positives: 3.

### Paragraph 4

Means of ‘managing’ which reduce villagers' direct dependence on local forest-resource use have also been suggested in this chapter. These include income-earning opportunities that do not draw on forest resources, such as in trade, and the substitution of imported food and goods for items produced using forest resources. They include ‘migrating’ economically, if not physically, away from the local forest-based economy, such as by securing remittances from urban contacts, by fosterage and by educating children. They also include involvement with non-local people to use forest resources in new ways, such as in diamond-digging, commercial hunting to supply long-distance bushmeat markets, and employment in logging companies. The balance between these various sources of livelihood, and the kinds of resource management involved in each, play a major role in shaping current and future pressures on the forest and forest-edge environments. In the conclusion we shall see some of the ways that Gola forest villagers reflect on such alternatives and their implications for forest futures which are at once social and environmental.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 121 | THEORETICAL ORIENTATION | x | x |
| 185 | CULTURAL GOALS | x | x |
| 311 | LAND USE | x |  |
| 434 | INCOME AND DEMAND | x | x |
| 464 | LABOR SUPPLY AND EMPLOYMENT | x | x |

Hits: 4 of 4 actual codes predicted · False positives: 1.

## nf12 — An inquiry into the political and economic structures of the Alexis Band of Wood Stoney Indians, 1880-1964

**SECTION:** CHAPTER VI SUMMARY AND CONCLUSIONS / Cultural Position of the Alexis Stoney  (8 paragraphs) — hdocid nf12-002

### Paragraph 1

Contributions to the literature about northern Plains Siouan speaking Stoney Assiniboine groups by such early writers as Denig, ^1 and Lowie, ^2 and the more recent work of Jenness, ^3 and Kennedy, ^4 have tended to develop a widely held view on the part of many anthropologists that the Assiniboine of the Plains and all Stoneys in the Alberta area were relatively identical during the historic period. Thus all Stoney and Assiniboine are generally regarded as having had a Plains horse nomadic buffalo hunting cultural ecological adaptation. Kennedy, for example, concludes that after 1750 the Stoney and Assiniboine were a typical Plains tribe whose existence depended “solely on the migratory buffalo.” ^5 ^1 ^2 ^3 ^4 ^5

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 101 | IDENTIFICATION | x |  |
| 114 | REVIEWS AND CRITIQUES | x | x |
| 171 | COMPARATIVE EVIDENCE | x | x |
| 174 | HISTORICAL RECONSTRUCTION | x |  |
| 221 | ANNUAL CYCLE | x |  |
| 231 | DOMESTICATED ANIMALS | x |  |
| 262 | DIET | x |  |

Hits: 2 of 2 actual codes predicted · False positives: 5.

### Paragraph 2

My examination of evidence gleaned from available historical sources and Alexis Stoney informants, bearing on the distribution, movements, and nature of early Siouan pedestrian Assiniboine peoples moving into the Rocky Mountain foothill region west of Edmonton, leads me to conclude that the currently prevailing view of their culture during the historic period is grossly incorrect. Sometime prior to 1800 and the appearance of the horse in significant numbers in this part of the northern Plains, small bands of Siouan speakers now known as Stoney moved into this fur rich region west of Edmonton. It seems certain that they established themselves along both banks of the Athabasca and North Saskatchewan Rivers and achieved a distinctive adaptation closely linked with the fur trade. I have termed this distinctive adaptation Wood as opposed to Plains Stoney. It should be clear here that there has been no relevant archaeological research in this area from which might be drawn information beyond that now available from historical sources and informants.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 101 | IDENTIFICATION | x | x |
| 105 | CULTURE SUMMARY | x |  |
| 114 | REVIEWS AND CRITIQUES | x | x |
| 174 | HISTORICAL RECONSTRUCTION | x | x |

Hits: 3 of 3 actual codes predicted · False positives: 1.

### Paragraph 3

The Wood Stoney were trappers, hunters, and fishers exploiting an ecotone on the northwestern edge of the Plains. The area is also a cultural shatterbelt of sorts in which the predecessors of the modern Alexis Wood Stoney came into contact with Plains, Subarctic, Plateau, and mountain Indian groups—many of which have left some cultural impact on the Wood Stoney. Although the wood buffalo was present in this area, it appears that they were never available in numbers sufficient to provide the basis for development of a pastoral nomadic economy. The Plains buffalo ranged north and west to the Edmonton area but its appearance this far north was quite variable and uncertain. This fact, and the presence of more militant Plains Indian groups such as the Blackfoot, Blood, Piegan, and Sarcee, probably discouraged regular movements onto the Plains by the Wood Stoneys.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 101 | IDENTIFICATION |  | x |
| 105 | CULTURE SUMMARY | x |  |
| 131 | LOCATION |  | x |
| 174 | HISTORICAL RECONSTRUCTION | x | x |

Hits: 1 of 3 actual codes predicted · False positives: 1.

### Paragraph 4

Horses would be a requisite to long-range buffalo hunting forays on the Plains by the Wood Stoneys, and it seems that those of this area had few. The horses they did have were generally in poor condition and unfit for long journeys to and from the Plains. It was difficult to care for and keep horses well nourished in the forage poor bush through the trapping season. Other factors complicating development and maintenance of good horse herds by these Wood Stoneys arise with respect to movement of men and horses, but particularly the horses, across the broad, cold, and swift-flowing Athabasca and North Saskatchewan Rivers in order to reach the Plains.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 105 | CULTURE SUMMARY | x |  |
| 174 | HISTORICAL RECONSTRUCTION | x | x |
| 221 | ANNUAL CYCLE | x |  |
| 231 | DOMESTICATED ANIMALS | x |  |
| 492 | ANIMAL TRANSPORT |  | x |

Hits: 1 of 2 actual codes predicted · False positives: 3.

### Paragraph 5

All of these factors probably marshalled against a shift to the Plains by these Wood Stoneys west of Edmonton, although some farther south along the foothills approximately in the area of modern Calgary apparently did make the important shift onto the Plains and became horse nomadic buffalo hunters. This may have occurred by about 1790 for the Bearspaws Band, and perhaps also for the Strongwoods Assiniboine, although the latter may not have been a Wood Stoney group. Both bands fit the rudimentary conception for the “typical” Plains Indian society of the historic period. Both the Plains buffalo and horses were far more numerous and accessible at this point on the northwestern Plains periphery.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 105 | CULTURE SUMMARY | x |  |
| 131 | LOCATION |  | x |
| 174 | HISTORICAL RECONSTRUCTION | x | x |
| 433 | PRODUCTION AND SUPPLY |  | x |

Hits: 1 of 3 actual codes predicted · False positives: 1.

### Paragraph 6

I do not intend here, nor do I have sufficient information, to do more than indicate the existence of the proposed Wood as opposed to Plains Stoney-Assiniboine cultural ecological adaptation. This differentiation seems largely related to different environmental conditions encountered by various groups in this area during the historic period. The problem involved is comparable to that of the Dakota or Sioux, which Howard holds may be divided into three divisions, each having achieved a special adaptation to a particular environment. ^6 The rather skeletal dichotomy I propose may be too simple, and further work among the Stoneys of Alberta may lead to information warranting perhaps three or four useful cultural ecological distinctions. Only future research will resolve this issue. ^6

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 101 | IDENTIFICATION | x |  |
| 105 | CULTURE SUMMARY | x |  |
| 114 | REVIEWS AND CRITIQUES | x |  |
| 131 | LOCATION |  | x |
| 171 | COMPARATIVE EVIDENCE | x |  |
| 174 | HISTORICAL RECONSTRUCTION |  | x |
| 433 | PRODUCTION AND SUPPLY |  | x |

Hits: 0 of 3 actual codes predicted · False positives: 4.

### Paragraph 7

In view of the foregoing, it is evident that the Stoneys from which the modern Alexis band originated are best conceived of as small bands of largely pedestrian hunters, trappers, and fishers of a forest and lake ecotone and cultural shatterbelt. They tended to be organized into small patrilocal camp groups during most of the year, and loosely allied bands far smaller in population when contrasted with estimates for their Plains nomadic pastoralist counterparts.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 101 | IDENTIFICATION |  | x |
| 105 | CULTURE SUMMARY | x | x |
| 174 | HISTORICAL RECONSTRUCTION | x |  |
| 221 | ANNUAL CYCLE |  | x |

Hits: 1 of 3 actual codes predicted · False positives: 1.

### Paragraph 8

The camp unit was the primary production and consumption unit, each camp tending to be politically autonomous, and camp and inter-camp political organization was informal. There was no primarily politically oriented concrete structure. Leaders of several camps, a small band, were limited in their power, which probably reached its apex during the summer encampments. It is also possible and even probable, although my data only hints at this, that the Wood Stoney band leader owed some of his influence to his reputation as an effective trading chief.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 101 | IDENTIFICATION |  | x |
| 105 | CULTURE SUMMARY | x | x |
| 174 | HISTORICAL RECONSTRUCTION | x |  |
| 221 | ANNUAL CYCLE |  | x |
| 621 | COMMUNITY STRUCTURE | x |  |

Hits: 1 of 3 actual codes predicted · False positives: 2.

## mj04 — The Arab of the desert : a glimpse into Badawin life in Kuwait and Sau'di Arabia

**SECTION:** Part One / CHAPTER XLI The Sulubba / THE RABÁBAH  (3 paragraphs) — hdocid mj04-001

### Paragraph 1

The rabábah, or Badawin guitar, was once universally used among the Badawin tribes of Arabia. The minstrels were men of an inferior tribe, or of the servant class, such as Sanas and the like. With the spread of Wahhabism and the rise of the fanatical 'Ikhwan in 1919, the decree went forth that the rabábah was a sign of sin, and it was banned entirely. The Northern Shammar, Dhafir and 'Anizah tribes, who were scarcely affected by the puritan revival, still continued to use the rabábah freely, getting the servants and retainers to play for them, and with the collapse of 'Ikhwanism after the rebellion and defeat of Faisal al Duwish in 1930, listening to the rabábah and the singing of war and love songs by the Sana sections of the tribes, appears to be once again slowly taking hold. To-day, 1937, some of the Rashaida(■) Hirshan and 'Awazim are definitely taking to it again. The Sulubba and Kauliyah (Gipsies of Iraq) have always used it, even in the days of ultra-Wahhabism, and perhaps this is why it was considered the instrument of the low-born.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 534 | MUSICAL INSTRUMENTS | x | x |
| 535 | DANCE | x |  |
| 554 | STATUS, ROLE, AND PRESTIGE | x |  |

Hits: 1 of 1 actual codes predicted · False positives: 2.

### Paragraph 2

To accompany women and girls dancing, only a chorus of women singers is used, never the rabábah. Men never dance to the tune of the instrument, but only sing to it, either singly or in unison, repeating the words of the soloist, and keeping time by clapping their hands.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 533 | MUSIC | x |  |
| 534 | MUSICAL INSTRUMENTS | x | x |
| 535 | DANCE | x | x |
| 554 | STATUS, ROLE, AND PRESTIGE |  | x |

Hits: 2 of 3 actual codes predicted · False positives: 1.

### Paragraph 3

It is definitely considered undignified for a pure-bred Arab to play the rabábah, and must always be so.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 534 | MUSICAL INSTRUMENTS | x | x |
| 535 | DANCE |  | x |
| 554 | STATUS, ROLE, AND PRESTIGE | x | x |

Hits: 2 of 3 actual codes predicted · False positives: 0.

## fy08 — The Tanala, a hill tribe of Madagascar

**SECTION:** THE TANALA, A HILL TRIBE OF MADAGASCAR / V. ECONOMIC LIFE / PREPARATION OF FOOD / FIRE MAKING  (7 paragraphs) — hdocid fy08-001

### Paragraph 1

The Tanala make fire by the following three methods: the saw, the drill in two forms, and flint and steel. According to native traditions the saw, called didiafo, is the oldest and was the only method employed by the aborigines of the Tanala country. At the present time it is rarely used and seems to be unknown to the other tribes of Madagascar. The apparatus is made from very dry bamboo, a rounded section being employed as the bed and a straight, flat piece as the saw. A shallow transverse groove is cut in the round side of the bed, without piercing its inner surface. The operator squats on the ground holding the bed in front of and parallel to his body with his feet. The saw is grasped by both ends and its edge inserted in the groove. It is then drawn rapidly back and forth, at right angles to the body (Fig. 4, g). Charred cotton, as tinder, is placed under and around the bed. The spark may be caught either in the hollow under the bed, when the saw pierces its lower surface, or at the side when the glowing dust runs down. Operators seemed to have a good deal of difficulty in catching the spark, but this was probably due to lack of familiarity with the apparatus. The method is fairly rapid, the saw beginning to smoke in about thirty seconds.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 372 | FIRE | x | x |

Hits: 1 of 1 actual codes predicted · False positives: 0.

### Paragraph 2

The simpler form of fire drill (Fig. 4, h) has a rectangular bed about ¾ inch wide by ½ inch thick and 1 foot to 18 inches long. A slight depression is cut in the upper surface with a notch or groove running from it to the edge. The drill is about 14 inches long and ½ inch in diameter. It is made of hard wood while the bed is of soft wood. The operator squats, holding the bed with his feet or having it held by an assistant. He inserts the point of the drill in the depression in the bed and twirls the shaft rapidly between his palms. He begins near the top of the drill and, as he twirls, brings his hands down, then lifts them to the top again. Considerable pressure is exerted. Cotton tinder is piled beside the bed against the notch. Wood dust ignited by the friction is forced out through the notch into the tinder. Fire is made in from forty seconds to a minute. This method is used by all the Madagascar tribes with only slight variations.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 372 | FIRE | x | x |

Hits: 1 of 1 actual codes predicted · False positives: 0.

### Paragraph 3

Pump drills are also used for fire making, but are much rarer than the plain drills. The bed is like that just described. The drill shaft is made from soft wood with a hard wood head attached. The whorl is also of wood, in the shape of a figure eight. The pumping stick is usually 8 inches to 1 foot long. The cord attached to it passes over a notch in the tip of the drill shaft (Fig. 5, f). In fire making, one man holds the bed while another pumps the drill up and down with both hands. No pressure is applied to the upper end of the drill shaft. The time required to produce a spark is about the same as with the ordinary drill. Pump fire drills are also used by the Sihanaka and probably by other tribes, but they are nowhere common.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 372 | FIRE | x | x |
| 412 | GENERAL TOOLS | x |  |

Hits: 1 of 1 actual codes predicted · False positives: 1.

### Paragraph 4

At the present time flint and steel is by far the commonest fire making appliance throughout Madagascar. The Tanala say that the historic clans, as distinct from the aborigines, have always used it. The striker is shaped somewhat like an adze blade, the narrow upper end being bent down over the back to form a grip (Fig. 4, d1). A variety of stones is used: flint, chalcedony, jasper and even quartz crystal. The tinder is made from charred cotton or cotton cloth. It is carried in the tip of a horn, the open end being provided with a tight stopper of gourd shell (Fig. 4, d3). The entire outfit is usually carried in a small box of wood or rawhide, in a horn, or in a hollow hide belt. Tinder horns and belts are used by all the Madagascar tribes, but the wooden tinder boxes appear to be limited to the Tanala. Nearly all of them are manufactured by the Zafimaniry division, which trades them to the neighboring clans. At least 80 per cent of the boxes conform to a single pattern (Fig. 5, b). The body has a long oval section, with flat, squarely cut ends. It is hollowed from a single block of rather soft, light colored wood which exhibits a beautiful curling grain. The bottom, in the form of a long pointed oval, is made separately and inset, being held in place by pins of wood, bone or horn. It is usually of red wood. The cover is made from a moderately hard red wood and fits closely over a sort of sleeve or projection, so that the joint is almost watertight. Its outer surface is flush with the body. It is surmounted by a comb, the width of the flat ends of the box, which is notched along the upper edge. Holes are pierced horizontally through the ends of the body a short distance below the lid and through the base of the comb near either end. A long cord passing through these holds the cover on and also serves for suspension. In a second and much rarer type one side of the box is flat and the other convex (Fig. 5, c). The lid may have a low comb along the flat side, but usually has a low flange running across the flat side and extending forward for a short distance on the curved side, at either end. A single double box of this type was collected and a modification in which the body of the box was crescent shaped in section was seen. In a third type, which is extremely rare, the body of the box is flat on one side and convex on the other and tapers to a point at the bottom. Tinder boxes are rarely if ever carved, but those of the first type are sometimes decorated with inset studs of metal or bone. They are always finely finished and with use and repeated oiling acquire a high polish. When traveling they are worn hanging on the breast or, rarely, at the belt.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 372 | FIRE | x | x |
| 415 | UTENSILS | x | x |
| 438 | INTERNAL TRADE | x | x |
| 531 | DECORATIVE ART |  | x |

Hits: 3 of 4 actual codes predicted · False positives: 0.

### Paragraph 5

Rawhide tinder boxes are rarely used by the northern Tanala, only two examples having been seen. They are made from long, rectangular strips of hide with the hair removed. While the hide is still soft, two-thirds of the strip is folded back upon itself and sewn together along the edges with a thong, forming a pouch. The remaining third is then bent down over this as a cover. Slits are cut in the rear of the pouch, or strips are sewn on, so that it can be strung on the belt. The flap is usually decorated with tooled designs (Fig. 5, a). This type seems to be peculiar to the Tanala, but hide tinder boxes of a different form are used by the Bara and other southern tribes.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 281 | WORK IN SKINS | x | x |
| 415 | UTENSILS | x | x |
| 531 | DECORATIVE ART | x | x |

Hits: 3 of 3 actual codes predicted · False positives: 0.

### Paragraph 6

The horn tinder boxes are made from sections of cow horn 8 to 18 inches long. The edges of the open end are cut off smoothly except at the inside of the horn's curve, where a rectangular projection about 1 inch wide and ½ inch to 1 inch long is left. An oval piece of wood, which has on one side a projection corresponding to that on the horn, is fitted into the opening and fastened solidly with wooden or horn pins. In the center of this there is an oval opening about 1 inch in diameter which is closed by a close fitting cork or stopper. This stopper is attached to the projection on the side of the horn by a short piece of cord. Heavier cords are attached to the horn at both ends so that the whole can be worn as a belt, the curve of the horn fitting against the wearer's side. The cords are of braided raffia (Fig. 5, e). Such boxes are rarely decorated but in one specimen the end of the horn has been cut into a series of knobs. Similar boxes are used by practically all the Madagascar tribes.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 321 | BONE, HORN, AND SHELL TECHNOLOGY | x | x |
| 415 | UTENSILS | x | x |
| 531 | DECORATIVE ART |  | x |

Hits: 2 of 3 actual codes predicted · False positives: 0.

### Paragraph 7

The hollow belts are made from the skin of a cow's tail drawn off whole. The skin is dried and flattened, the hair being left on. The smaller end is sewn up and a loop of hide attached. The larger end is pierced near either side. A wide thong is split for several inches and the ends of the split portion passed through the holes in the belt and fastened to either end of a flat strip of bamboo. The other end of this thong and the loop sewn to the smaller end of the tail are tied together. The tension draws the strip of bamboo against the belt, closing the opening (Fig. 5, d). Similar belts are used by the southeast coast tribes but are rare or lacking elsewhere in Madagascar.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 281 | WORK IN SKINS | x | x |
| 321 | BONE, HORN, AND SHELL TECHNOLOGY | x |  |
| 372 | FIRE | x | x |
| 412 | GENERAL TOOLS |  | x |
| 415 | UTENSILS |  | x |

Hits: 2 of 4 actual codes predicted · False positives: 1.

## nt18 — Barter, gift, or violence: an analysis of Tewa inter tribal exchange

**SECTION:** BARTER, GIFT, OR VIOLENCE: AN ANALYSIS OF TEWA INTERTRIBAL EXCHANGE / III  (6 paragraphs) — hdocid nt18-017

### Paragraph 1

The Tewa are a self-defined linguistic group composed of six pueblos—San Juan, Santa Clara, San Ildefonso, Pojoaque, Nambé, and Tesuque—located in north central New Mexico along the Rio Grande and several of its tributaries (Fig. 1). These communities have been visited and studied by anthropologists for a number of years (e.g., Harrington, 1916; Parsons, 1929), but it is only through the more recent work of Dozier (1960, 1961, 1970) and Ortiz (1965, 1969) that we are beginning to understand the underlying tenets of Tewa ritual structure and world view.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 101 | IDENTIFICATION | x | x |
| 131 | LOCATION | x | x |
| 184 | CULTURAL PARTICIPATION | x |  |

Hits: 2 of 2 actual codes predicted · False positives: 1.

### Paragraph 2

The local ecosystems vary slightly from pueblo to pueblo, but basically they are similar. During the period under consideration, the nineteenth century and first decade of the twentieth, individual village populations never exceeded 400 residents. The subsistence economy was based on the irrigated production of native corn, beans, squash, and kitchen garden crops as well as on gathered greens, roots, seeds, and fruits. Hunting contributed significantly to the diet during the winter when deer and elk were hunted locally, while early fall hunting expeditions to the plains of eastern New Mexico brought back antelope and buffalo meat.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 131 | LOCATION | x |  |
| 161 | POPULATION | x |  |
| 184 | CULTURAL PARTICIPATION |  | x |
| 222 | COLLECTING | x | x |
| 224 | HUNTING AND TRAPPING | x | x |
| 241 | TILLAGE | x | x |
| 262 | DIET | x |  |
| 433 | PRODUCTION AND SUPPLY | x | x |

Hits: 4 of 5 actual codes predicted · False positives: 3.

### Paragraph 3

Tewa food and craft production is embedded in the household which is often extended to include several generations. However, the vagaries of the Rio Grande environment are such that no matter how many hands may be available or how much land one may have inherited from his bilateral relations, natural disasters often limit productivity and result in both an unpredictably differential food supply for each family and a meager surplus for the village as a whole. Production variability of craft objects has a different basis. Craft items are not manufactured in every household. Some families make very few items, relying instead on bartering or borrowing for their needs. At the other extreme are the families in which one or more members produce a surplus intended for external consumption. Excepting ritual activities, most economic affairs are not coordinated by anyone except the household itself.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 430 | EXCHANGE AND TRANSFERS |  | x |
| 433 | PRODUCTION AND SUPPLY | x |  |
| 437 | EXCHANGE TRANSACTIONS | x |  |
| 592 | HOUSEHOLD | x | x |

Hits: 1 of 2 actual codes predicted · False positives: 2.

### Paragraph 4

At a higher level, each Tewa pueblo is organized into two dual divisions and several cross-cutting sodalities. For ritual and other purposes the head (cacique) of each moiety has ceremonial custodianship over the community for part of the year. With few exceptions, all members of the community belong to their father's moiety, and each cacique is assisted by a sodality composed of his own moiety's members. These moiety-based sodalities and the other associations organize dances and rituals of a specific nature and perform other explicit functions. The Bear curing societies, the Kossa and Kwirena clown societies, and the Hunt society have an obligation to assure the well-being of people, plants, and animals. The Scalp society, a warrior sodality, is linked to the Women's society which helps with the care of the scalps. It is through sodalities that all dances are performed and the symbols appropriate to each steadfastly maintained.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 575 | SODALITIES | x | x |
| 616 | MOIETIES | x | x |
| 622 | COMMUNITY HEADS | x | x |
| 624 | LOCAL OFFICIALS | x |  |
| 796 | ORGANIZED CEREMONIAL | x |  |

Hits: 3 of 3 actual codes predicted · False positives: 2.

### Paragraph 5

Facing the outside world are other organizations with political and protective functions. The most conspicuous person is a Spanish-imposed and American government-recognized governor who is selected annually. He has other assistants, but his decisions must be seconded by a village council. The other organization consists of the war captains, a native institution modified by Spanish authorities to assist them against other Indians, who are also appointed on an annual basis. They have the twofold task of defending the pueblo from outside attack as well as protecting all persons participating in rituals, public and secret.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 177 | ACCULTURATION AND CULTURE CONTACT | x | x |
| 622 | COMMUNITY HEADS | x | x |
| 623 | COMMUNITY COUNCILS | x | x |
| 624 | LOCAL OFFICIALS | x | x |
| 701 | MILITARY ORGANIZATION | x | x |

Hits: 5 of 5 actual codes predicted · False positives: 0.

### Paragraph 6

In a weak, diffuse polity with a low, differential household production, the ritual system helps to regulate many of the ecological variables of each Tewa ecosystem, and trade becomes a necessary safety valve for pueblo survival. But since each Tewa pueblo was part of a larger state hegemony—Spanish, Mexican, and American—which was unable to adequately protect individual pueblos, still less the traders going between communities, until the end of the last century, we begin to understand something about the difficulty of trade. The acuteness of this problem is evident in an examination of Tewa commerce with Plains Indians.

| OCM | Label | Predicted | Actual |
|---|---|---|---|
| 182 | FUNCTIONAL/ADAPTATIONAL INTERPRETATIONS | x | x |
| 430 | EXCHANGE AND TRANSFERS | x | x |

Hits: 2 of 2 actual codes predicted · False positives: 0.

## Legend

- **OCM** — Outline of Cultural Materials subject code.
- **Label** — code name from the published OCM codebook.
- **Predicted** — the model assigned this code to the paragraph.
- **Actual** — the original analysts assigned this code to the paragraph.
- **Hits: k of m** — k of the m actual codes for the paragraph were also
  predicted by the model.
- **False positives: f** — f predicted codes were *not* among the
  paragraph's actual codes. A false positive may still be a defensible code
  for the passage (or may indicate a real disagreement between the model
  and the analysts) — that judgment is exactly what this review asks about.

## Results summary

51 paragraphs across 10 sections (3–8 paragraphs, ≥3 distinct codes each, one section per document): precision 0.647, recall 0.724, F1 0.683, exact-set match 0.196.

Patterns worth noting (from a first read of this sample; the sections are
identical to those in the v5 model's review sample, `eHRAF_analysis_v5_a.md`,
so each paragraph can be compared side by side across the two reports):

- Recall is visibly higher than v5. Paragraphs the v5 model gutted are now
  largely complete: the ef05 Blood-revenge section recovers 10 of 10 actual
  codes across its three paragraphs, the Tewa political-organization
  paragraph hits 5 of 5, and the tinder-box 415 UTENSILS codes that v5 missed
  in the Tanala fire-making section are now caught. The Mende CONCLUSION
  section's referential paragraphs (121 THEORETICAL ORIENTATION) are again
  recovered, now together with more of their substantive companions.
- The section-header prior still anchors recurring codes: 824 ETHNOBOTANY is
  held across all six catalogue paragraphs, 174 HISTORICAL RECONSTRUCTION
  across the Stoney "Cultural Position" section, 756 SHAMANS / 776 SPIRITS
  across the Lengua shaman section, 372 FIRE through the Tanala methods.
- A new weakness is over-generation on *thin* paragraphs. In the st13
  botanical catalogue, bare citation stubs ("Chenopodium ambrosioides
  Linnaeus, Sp. Pl. 219. 1753.", "Specimen citation: 3364.") each attract an
  inferred 278 PHARMACEUTICALS that the analysts reserved for the one
  substantive paragraph; the Stoney literature-review paragraphs draw wide
  near-miss lists (221 ANNUAL CYCLE, 231 DOMESTICATED ANIMALS, 262 DIET) for
  practices the text merely *discusses*. Where v5 under-generated, this
  model can code text *about* a topic as if the topic were *enacted* in it.
- Related-code substitutions now appear as false positives rather than
  misses: 412 GENERAL TOOLS shows up beside (no longer instead of) 372 FIRE
  for fire-drill parts, and 105 CULTURE SUMMARY stands in for 101
  IDENTIFICATION / 174 HISTORICAL RECONSTRUCTION in the Stoney section.
