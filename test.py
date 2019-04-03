from pymetamap import MetaMap
mm = MetaMap.get_instance('/home/noway/Facultate/Licenta/public_mm/bin/metamap18')
sents = ['Heart Attack', 'John had a huge heart attack']
concepts, error = mm.extract_concepts(sents)
for concept in concepts:
    print(concept.preferred_name)
