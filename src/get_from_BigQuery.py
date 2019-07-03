import pandas as pd
import re
from src.my_lib import get_data, get_first_claim

QUERY = ('''
#standardSQL

/*This sub-table returns the first OA for an application with:
       - a 101 rejection
       - no claims allowed (almost certainly a 101 applied to claim 1)
       - within art unit 363 (e-commerce) which is one of the hardest art units */
with 
  FIRST_OA as (
    SELECT  
      T1.app_id, 
      T1.mail_dt,
      T1.ifw_number
    FROM `patents-public-data.uspto_oce_office_actions.office_actions` T1
    INNER JOIN
      (
        SELECT app_id, MIN(mail_dt) as date
        FROM `patents-public-data.uspto_oce_office_actions.office_actions` 
        GROUP BY 
          app_id
       ) T2 
       ON T1.app_id = T2.app_id AND T1.mail_dt = T2.date
    WHERE 
      rejection_101 = '1'
      AND art_unit LIKE '362%'
      AND allowed_claims = '0'
    ORDER By 
      app_id), 

  /* For testing purposes I am only lookiong at patents that were granted between 2015 and 2018*/
  GRNT as(
  SELECT 
      application_number, 
      grant_date
  FROM `patents-public-data.patents.publications_201809`
  WHERE 
    grant_date > 20150000
    AND grant_date < 20180000
    AND country_code = 'US'),

  /* subtable for retreiving the earliest publication num & claims */
  AS_FILED as (
    SELECT 
        publication_date,
        PUB.application_number,
        PUB.publication_number,
        claims.text as filed_claims
    FROM `patents-public-data.patents.publications_201809` as PUB,
        UNNEST (claims_localized) as claims
    JOIN GRNT
      ON GRNT.application_number = PUB.application_number
    INNER JOIN
      (SELECT 
        application_number, 
        MIN(publication_date) as date
       FROM `patents-public-data.patents.publications_201809`
       GROUP BY
        application_number) T3
      ON 
        PUB.application_number = T3.application_number 
        AND PUB.publication_date = T3.date
    ORDER BY
      PUB.application_number),

  /* subtable for retreiving the granted publication num & claims */
  AS_GRANTED as(
     SELECT 
        publication_date,
        PUB.application_number,
        PUB.publication_number,
        claims.text as granted_claims
    FROM `patents-public-data.patents.publications_201809` as PUB,
        UNNEST (claims_localized) as claims
    JOIN GRNT
      ON GRNT.application_number = PUB.application_number
    INNER JOIN
      (SELECT 
        application_number, 
        MAX(publication_date) as date
       FROM `patents-public-data.patents.publications_201809`
       GROUP BY
        application_number) T3
      ON 
        PUB.application_number = T3.application_number 
        AND PUB.publication_date = T3.date
    ORDER BY
      PUB.application_number)

/* FINAL QUERY */
SELECT 
    OA.app_id, 
    APP.application_number,
    AS_FILED.publication_number as first_pub_no,
    AS_GRANTED.publication_number as granted_pub_no,
    OA.art_unit, 
    OA.rejection_101, 
    OA.rejection_102, 
    OA.rejection_103, 
    OA.allowed_claims, 
    OA.mail_dt, 
    OA.uspc_class,
    AS_FILED.filed_claims,
    AS_GRANTED.granted_claims
FROM `patents-public-data.uspto_oce_office_actions.office_actions`as OA
INNER JOIN FIRST_OA                      
  ON OA.ifw_number = FIRST_OA.ifw_number
JOIN `patents-public-data.uspto_oce_office_actions.match_app`as APP
  ON OA.app_id = APP.app_id
JOIN AS_FILED
  ON APP.application_number = AS_FILED.application_number
JOIN AS_GRANTED
  ON APP.application_number = AS_GRANTED.application_number
ORDER BY
  APP.application_number
''')

get_data(QUERY, 'art_unit_362.pkl')