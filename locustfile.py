from locust import HttpUser, task, between

class LoadTestUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def search(self):
        self.client.post(
            url="/multi_search?x-typesense-api-key=09EqyvhZTK4bRlTGe9WrYNBIi5gUuosl",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json"
            },
            json={
                "searches": [
                    {
                        "collection": "NET6_product_blue",
                        "filter_by": "",
                        "q": "sua tươi",
                        "query_by": "ecom_product_name,com_manufacturer_name,com_category_name,keywords,description",
                        "query_by_weights": "10,10,10,2,1",
                        "sort_by": "_text_match:desc,popularity:desc",
                        "text_match_type": "sum_score",
                        "stopwords": "stopword_set1",
                        "page": 1,
                        "per_page": 36,
                        "facet_by": "*",
                        "max_facet_values": 10,
                        "include_fields": "",
                        "exclude_fields": "is_deleted,erp_product_name,erp_manufacturer_name,erp_manufacturer_id ,embedding_reason_of_failure,created_date,auto_2_embedding,text_embedding,img_embedding,description,promotion_name_unaccent,com_manufacturer_name_unaccent,ecom_product_name_unaccent,com_category_name_unaccent,highlights,product_attributes,image_embedding,keywords,product_image_trans,checksum,embeddingerp_product_id,erp_category_name,erp_category_id, com_manufacturer_image_square_url",
                        "exhaustive_search": False,
                        "num_typos": 0,
                        "prioritize_token_position": False
                    }
                ]
            }
        )