SYS_PROMPT = """You'll generate sentence-semantic representation pairs about animals similar to the given examples below. \
Try to make the examples as diverse as possible. \
Try to make the scenes as general and high-level as possible (don't use adjectives). \n

\n
SENTENCE: the butterfly landed gracefully on the bright flower \n
SEM_REP: ,small,animate,insect,fly,plant \n
\n
SENTENCE: a fierce lion chases its prey \n
SEM_REP: ,animate,predator,hunt,move \n
\n
SENTENCE: the tiny ant is carrying food back to the colony \n
SEM_REP: ,small,animate,move,group \n
\n
SENTENCE: the blue whale swims in the deep ocean \n
SEM_REP: ,large,animate,mammal,water,move \n
\n
SENTENCE: the spider carefully weaves its web \n
SEM_REP: ,small,insect,animate,create \n
\n
SENTENCE: several colorful parrots are eating seeds \n
SEM_REP: ,animate,bird,small,group,food \n
\n
SENTENCE: a honeybee buzzes between flowers in the garden \n
SEM_REP: ,small,animate,insect,move,air,plant \n
\n
SENTENCE: the old turtle basks quietly in the sunlight \n
SEM_REP: ,animate,reptile,slow,still \n
\n
SENTENCE: an octopus changes colors rapidly \n
SEM_REP: ,animate,water,change \n
\n
"""
