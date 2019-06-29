import re
import requests
from main import Preprocesor
from multiprocessing.pool import ThreadPool
from os.path import exists


def translate_umls_cuis(ticket_granting_ticket, begin, end, filepath=r'./raw_features/old/semantic_types_ADE.txt',
                        new_filepath=r'./raw_features/old/semantic_types_translated_cuis_ADE.txt'):
    if not exists(new_filepath):
        with open(new_filepath, 'w'):
            pass

    with open(filepath, 'r') as fdr:
        with open(new_filepath, 'a') as fdw:
            lines = fdr.readlines()
            count = begin
            len_lines = len(lines)
            for line in lines[begin:end]:
                count += 1
                sem_types = re.findall(r'[a-zA-Z]{2,}', line)
                cuis = re.findall(r'C\d+', line)
                cui_concept_names = []
                for cui in cuis:
                    concept_name = get_concept_name_using_cui(cui, ticket_granting_ticket)
                    cui_concept_names.append(concept_name)
                text = " ".join(sem_types + cui_concept_names)
                fdw.write(str(count - 1) + '|' + Preprocesor.get_basic_preprocessig(text) + '\n')
                print(count, '/', len_lines)

    print("UMLS semantic types and cui concept names features save in:", new_filepath)


def get_file_len(filepath=r'./raw_features/old/semantic_types_ADE.txt'):
    with open(filepath, 'r') as fd:
        return len(fd.readlines())


def get_umls_service_ticket(ticket_granting_ticket):
    url = r'https://utslogin.nlm.nih.gov/cas/v1/api-key/'
    data = {'service': r'http://umlsks.nlm.nih.gov'}
    r = requests.post(url=url + ticket_granting_ticket, data=data)
    return r.content.decode()


def get_concept_name_using_cui(cui, ticket_granting_ticket):
    service_ticket = get_umls_service_ticket(ticket_granting_ticket)
    url = r'https://uts-ws.nlm.nih.gov/rest/content/current/CUI/' + cui + r'?ticket=' + service_ticket
    r = requests.get(url)
    return r.json()['result']['name']


def thread_function(ticket_granting_ticket, begin, end):
    translate_umls_cuis(ticket_granting_ticket, begin, end)
    print("Finish work", begin, "-", end)


def use_threads_for_multiple_reqs(ticket_grant_ticket, processes=101):
    pool = ThreadPool(processes=processes)
    async_result = []
    max_len = get_file_len()
    step = max_len // (processes - 1)
    print("Max len", max_len)
    begin = 0
    end = step
    for i in range(processes - 1):
        async_result.append(pool.apply_async(thread_function, (ticket_grant_ticket, begin, end)))
        begin += step
        end += step

    async_result.append(pool.apply_async(thread_function, (ticket_grant_ticket, begin, max_len)))

    for i in range(processes):
        async_result[i].get()


def sort_create_final_file(old_filepath=r'./raw_features/old/semantic_types_translated_cuis_ADE.txt',
                           new_filepath=r'./raw_features/semantic_types_cuis_name_ADE.txt'):
    with open(old_filepath, 'r') as fd:
        data = fd.readlines()
        data = sorted(data, key=lambda x: int(re.findall(r'^\d+', x)[0]))
    with open(new_filepath, 'w') as fd:
        for d in data:
            fd.write(d.split("|")[1].rstrip() + '\n')
    print("Final umls features save in", new_filepath)


def main():
    # use_threads_for_multiple_reqs(ticket_grant_ticket=?)
    sort_create_final_file()


if __name__ == '__main__':
    main()
