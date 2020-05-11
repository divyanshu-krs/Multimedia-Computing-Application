from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy import sparse

def relevance_feedback(vec_docs, vec_queries, gt, sim, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    #sim_matrix = cosine_similarity(vec_docs, vec_queries)
    top_n_relevant_doc = []
    for i in range(sim.shape[1]):
            ranked_documents = np.argsort(-sim[:, i])
            top_n_relevant_doc.append(ranked_documents[:n])
            #print ('Query:', i+1, 'Top relevant n documents:', ranked_documents[:n] + 1)
    real_rel = []
    non_rel = []
    
    for i in range(len(top_n_relevant_doc)):
        
        query = i
        curr = top_n_relevant_doc[query]
        a = []
        b = []
        
        for j in range(len(gt)):
            #print('gt[j][0]',gt[j][0])
            #print('query number', query)
            if (gt[j][0] == query+1):
                
                
                if ( gt[j][1] not in list(curr)):
                    a.append(gt[j][1])
                else:
                    b.append(gt[j][1])
                   
        real_rel.append(b)
        non_rel.append(a)

    #print(real_rel)
    #print(non_rel)
        
    alpha = 0.1
    beta = 1

    new_vec_queries = np.zeros([30,10625])
    
    for i in range(30):
        query = vec_queries.toarray()[i]
        rel_doc = real_rel[i]
        non_doc = non_rel[i]

        ##
        weight_up_rel = np.zeros([10625,])
        for j in rel_doc:
            weight_doc = vec_docs.toarray()[j-1]
            weight_up_rel += weight_doc
            
        weight_up_rel = alpha * weight_up_rel

        ##


        ##
        weight_up_non = np.zeros([10625,])
        for k in non_doc:
            doc_w = vec_docs.toarray()[k-1]
            weight_up_non += doc_w

        weight_up_non = beta * weight_up_non
        ##

        new_vec_queries[i] = query + weight_up_rel + weight_up_non

    new_vec_queries = sparse.csr_matrix(new_vec_queries)
    sim =  cosine_similarity(vec_docs, new_vec_queries)   
            
    rf_sim = sim # change
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim,gt,n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """

        
    top_n_relevant_doc = []
    for i in range(sim.shape[1]):
            ranked_documents = np.argsort(-sim[:, i])
            top_n_relevant_doc.append(ranked_documents[:n])
            #print ('Query:', i+1, 'Top relevant n documents:', ranked_documents[:n] + 1)
    real_rel = []
    non_rel = []
    
    for i in range(len(top_n_relevant_doc)):
        
        query = i
        curr = top_n_relevant_doc[query]
        a = []
        b = []
        
        for j in range(len(gt)):
            #print('gt[j][0]',gt[j][0])
            #print('query number', query)
            if (gt[j][0] == query+1):
                
                
                if ( gt[j][1] not in list(curr)):
                    a.append(gt[j][1])
                else:
                    b.append(gt[j][1])
                   
        real_rel.append(b)
        non_rel.append(a)

    #print(real_rel)
    #print(non_rel)
        
    alpha = 0.1
    beta = 1

    new_vec_queries = np.zeros([30,10625])
    
    for i in range(30):
        query = vec_queries.toarray()[i]
        rel_doc = real_rel[i]
        non_doc = non_rel[i]

        ##
        weight_up_rel = np.zeros([10625,])
        for j in rel_doc:
            weight_doc = vec_docs.toarray()[j-1]
            weight_up_rel += weight_doc
            
        weight_up_rel = alpha * weight_up_rel

        ##

        ##
        weight_up_non = np.zeros([10625,])
        for k in non_doc:
            doc_w = vec_docs.toarray()[k-1]
            weight_up_non += doc_w

        weight_up_non = beta * weight_up_non
        ##

        new_vec_queries[i] = query + weight_up_rel + weight_up_non

    new_vec_queries = sparse.csr_matrix(new_vec_queries)


######## After Updating #########
    update_rank_doc = []
    for i in range(sim.shape[1]):
            ranked_documents = np.argsort(-sim[:, i])
            update_rank_doc.append(ranked_documents[:10])
            
    #print(update_rank_doc)
    up_rel = []
    up_non = []
    
    for i in range(len(update_rank_doc)):
        
        query = i
        curr = update_rank_doc[query]
        a = []
        b = []
        
        for j in range(len(gt)):
            #print('gt[j][0]',gt[j][0])
            #print('query number', query)
            if (gt[j][0] == query+1):
                
                
                if ( gt[j][1] not in list(curr)):
                    a.append(gt[j][1])
                else:
                    b.append(gt[j][1])
                   
        up_rel.append(b)
        up_non.append(a)


    
    all_rel_doc_tfidf = []
    
    all_rel_doc_index = []
    
    
    for i in up_rel:
        
        doc_tfidf = []
        index = []
        
        for doc_num in i:
            
            ini_v_d = vec_docs.toarray()[doc_num-1]
            v_d = np.sort(ini_v_d)[::-1]
            
            for u in range(10):
                tf = v_d[u]
                ind = list(ini_v_d).index(tf)
                index.append(ind)
                doc_tfidf.append(v_d[u])

        all_rel_doc_tfidf.append(doc_tfidf)
        all_rel_doc_index.append(index)
        

    final_vec_queries = np.zeros([30,10625])
    
    for i in range(30):
        
        query = new_vec_queries.toarray()[i]
        tfidf = all_rel_doc_tfidf[i]
        index = all_rel_doc_index[i]

        
        for j in range(len(index)):
            query[index[j]] += tfidf[j]
            
            
        final_vec_queries[i] = query

    final_vec_queries = sparse.csr_matrix(final_vec_queries)
                  
            
    
    sim =  cosine_similarity(vec_docs, final_vec_queries)  

    rf_sim = sim  # change
    return rf_sim
