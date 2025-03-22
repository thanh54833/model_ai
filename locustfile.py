import random

from locust import HttpUser, between, task


class LoadTestUser(HttpUser):
    wait_time = between(1, 5)

    keywords = [
        {"name": "sữa", "id": 0},
        {"name": "trà", "id": 1},
        {"name": "cà phê", "id": 2},
        {"name": "nước ngọt", "id": 3},
        {"name": "bánh mì", "id": 4},
        {"name": "kẹo", "id": 5},
        {"name": "sữa chua", "id": 6},
        {"name": "bia", "id": 7},
        {"name": "mì gói", "id": 8},
        {"name": "gạo", "id": 9},
        {"name": "dầu ăn", "id": 10},
        {"name": "nước mắm", "id": 11},
        {"name": "đường", "id": 12},
        {"name": "muối", "id": 13},
        {"name": "bột ngọt", "id": 14},
        {"name": "sữa bột", "id": 15},
        {"name": "bỉm", "id": 16},
        {"name": "khăn giấy", "id": 17},
        {"name": "nước rửa chén", "id": 18},
        {"name": "xà phòng", "id": 19},
        {"name": "kem đánh răng", "id": 20},
        {"name": "dầu gội", "id": 21},
        {"name": "sữa tắm", "id": 22},
        {"name": "nước xả vải", "id": 23},
        {"name": "bột giặt", "id": 24},
        {"name": "nước lau sàn", "id": 25},
        {"name": "tã giấy", "id": 26},
        {"name": "khăn ướt", "id": 27},
        {"name": "bàn chải", "id": 28},
        {"name": "thực phẩm chức năng", "id": 29},
        {"name": "vitamin", "id": 30},
        {"name": "mật ong", "id": 31},
        {"name": "nước ép", "id": 32},
        {"name": "trái cây", "id": 33},
        {"name": "rau củ", "id": 34},
        {"name": "thịt heo", "id": 35},
        {"name": "thịt gà", "id": 36},
        {"name": "cá", "id": 37},
        {"name": "tôm", "id": 38},
        {"name": "mực", "id": 39},
        {"name": "bánh kẹo", "id": 40},
        {"name": "snack", "id": 41},
        {"name": "bánh quy", "id": 42},
        {"name": "chocolate", "id": 43},
        {"name": "kẹo mút", "id": 44},
        {"name": "nước khoáng", "id": 45},
        {"name": "sữa tươi", "id": 46},
        {"name": "phô mai", "id": 47},
        {"name": "bơ", "id": 48},
        {"name": "kem", "id": 49},
        {"name": "yogurt", "id": 50},
        {"name": "nước tăng lực", "id": 51},
        {"name": "trà xanh", "id": 52},
        {"name": "cà phê sữa", "id": 53},
        {"name": "nước dừa", "id": 54},
        {"name": "sữa đậu nành", "id": 55},
        {"name": "bánh trung thu", "id": 56},
        {"name": "mứt", "id": 57},
        {"name": "hạt điều", "id": 58},
        {"name": "hạt dẻ", "id": 59},
        {"name": "ô mai", "id": 60},
        {"name": "nước tương", "id": 61},
        {"name": "tương ớt", "id": 62},
        {"name": "mayonnaise", "id": 63},
        {"name": "dầu hào", "id": 64},
        {"name": "gia vị", "id": 65},
        {"name": "bột chiên giòn", "id": 66},
        {"name": "bột nêm", "id": 67},
        {"name": "thịt bò", "id": 68},
        {"name": "trứng gà", "id": 69},
        {"name": "trứng vịt", "id": 70},
        {"name": "đậu hũ", "id": 71},
        {"name": "chả lụa", "id": 72},
        {"name": "xúc xích", "id": 73},
        {"name": "thịt nguội", "id": 74},
        {"name": "cá hộp", "id": 75},
        {"name": "thịt hộp", "id": 76},
        {"name": "mì Ý", "id": 77},
        {"name": "bún", "id": 78},
        {"name": "phở", "id": 79},
        {"name": "bánh canh", "id": 80},
        {"name": "nước lọc", "id": 81},
        {"name": "khăn mặt", "id": 82},
        {"name": "dép", "id": 83},
        {"name": "quần áo trẻ em", "id": 84},
        {"name": "đồ chơi", "id": 85},
        {"name": "sách", "id": 86},
        {"name": "vở", "id": 87},
        {"name": "bút", "id": 88},
        {"name": "hộp đựng thức ăn", "id": 89},
        {"name": "ly nhựa", "id": 90},
        {"name": "muỗng", "id": 91},
        {"name": "đĩa", "id": 92},
        {"name": "chén", "id": 93},
        {"name": "nồi", "id": 94},
        {"name": "chảo", "id": 95},
        {"name": "bếp ga", "id": 96},
        {"name": "bình nước", "id": 97},
        {"name": "hộp cơm", "id": 98},
        {"name": "túi đựng rác", "id": 99}
    ]

    @task
    def search(self):
        keyword = random.choice(self.keywords)["name"]
        # self.client.post(
        #     url="/multi_search?x-typesense-api-key=09EqyvhZTK4bRlTGe9WrYNBIi5gUuosl",
        #     headers={
        #         "Accept": "application/json",
        #         "Content-Type": "application/json"
        #     },
        #     json={
        #         "searches": [
        #             {
        #                 "collection": "NET6_product_blue",
        #                 "filter_by": "",
        #                 "q": keyword,
        #                 "query_by": "ecom_product_name,com_manufacturer_name,com_category_name,keywords,description",
        #                 "query_by_weights": "10,10,10,2,1",
        #                 "sort_by": "_text_match:desc,popularity:desc",
        #                 "text_match_type": "sum_score",
        #                 "stopwords": "stopword_set1",
        #                 "page": 1,
        #                 "per_page": 36,
        #                 "facet_by": "*",
        #                 "max_facet_values": 10,
        #                 "include_fields": "",
        #                 "exclude_fields": "is_deleted,erp_product_name,erp_manufacturer_name,erp_manufacturer_id ,embedding_reason_of_failure,created_date,auto_2_embedding,text_embedding,img_embedding,description,promotion_name_unaccent,com_manufacturer_name_unaccent,ecom_product_name_unaccent,com_category_name_unaccent,highlights,product_attributes,image_embedding,keywords,product_image_trans,checksum,embeddingerp_product_id,erp_category_name,erp_category_id, com_manufacturer_image_square_url",
        #                 "exhaustive_search": False,
        #                 "num_typos": 0,
        #                 "prioritize_token_position": False
        #             }
        #         ]
        #     }
        # )

        self.client.post(
            url="https://ecommerce-uat.concung.vn/api/se/product-search/search_text",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json"
            },
            json={
                "q": keyword,
                "page_index": 1,
                "page_size": 36,
                "filter_by": "",
                "sort_by": "",
                "facet_by": "",
                "is_search_for_ecom5": True,
                "is_return_filter": True
            }
        )
