"""50 экономных агрегатов по рискам моделей и GenAI-агентов.

Результат:
arnsdpsbx_t_team_oam_sva_2.pri_model_lib_analysis

Скрипт читает только текущие таблицы и только используемые столбцы,
агрегирует данные и не сохраняет персональные/текстовые детали.
"""

import os
import sys
from functools import reduce
from typing import List


os.environ["SPARK_MAJOR_VERSION"] = "3.5.1"
os.environ["SPARK_HOME"] = "/usr/sdp/current/spark3.5.1-client/"
os.environ["PYSPARK_DRIVER"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

sys.path.insert(0, "/usr/sdp/current/spark3.5.1-client/python/")
sys.path.insert(
    0,
    "/usr/sdp/current/spark3.5.1-client/python/lib/py4j-0.10.9.7-src.zip",
)

from pyspark import SparkConf, StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F


SRC_DB = "prx_pri_custom_ris_l_library_custom_risk_model_library"
OUT_TABLE = "arnsdpsbx_t_team_oam_sva_2.pri_model_lib_analysis"

OPEN_SLA_DAYS = 90
STALE_MONITORING_DAYS = 365
CRITICAL_SIGNIFICANCE = ("A", "B")
SHOW_ROWS = 300

conf = (
    SparkConf()
    .setAppName("PGS")
    .setMaster("yarn")
    .set("spark.executor.cores", "2")
    .set("spark.executor.memory", "6g")
    .set("spark.executor.memoryOverhead", "1g")
    .set("spark.driver.memory", "6g")
    .set("spark.driver.maxResultSize", "8g")
    .set("spark.shuffle.service.enabled", "true")
    .set("spark.hadoop.mapreduce.input.fileinputformat.input.dir.recursive", "true")
    .set("spark.dynamicAllocation.enabled", "true")
    .set("spark.dynamicAllocation.executorIdleTimeout", "120s")
    .set("spark.dynamicAllocation.cachedExecutorIdleTimeout", "600s")
    .set("spark.dynamicAllocation.initialExecutors", "3")
    .set("spark.dynamicAllocation.maxExecutors", "12")
    .set("spark.dynamicAllocation.shuffleTracking.enabled", "true")
    .set("spark.port.maxRetries", "150")
    .set("dfs.replication", "2")
    .set("spark.sql.parquet.writeLegacyFormat", "true")
    .set("spark.sql.parquet.compression.codec", "snappy")
    .set("parquet.block.size", str(128 * 1024 * 1024))
    .set("dfs.block.size", str(128 * 1024 * 1024))
    .set("spark.sql.session.timeZone", "Europe/Moscow")
)

spark = (
    SparkSession.builder
    .config(conf=conf)
    .enableHiveSupport()
    .getOrCreate()
)

# Закрывающие статусы задаются regex, чтобы покрыть русские и английские коды.
CLOSED_STATUS_RE = r"DONE|COMPLETE|COMPLETED|CLOSED|CANCEL|CANCELED|REJECT|ARCHIV|ЗАВЕРШ|ЗАКРЫТ|ОТМЕН|ОТКЛОН"
RED_RESULT_RE = r"RED|КРАСН"
OOT_RE = r"OUT[-_ ]?OF[-_ ]?TIME|\bOOT\b|ВНЕВРЕМ"


def _ts(name: str):
    """Разбирает timestamp/date и распространённые строковые форматы."""
    raw = F.col(name)
    s = raw.cast("string")
    return F.coalesce(
        raw.cast("timestamp"),
        F.to_timestamp(s, "dd.MM.yyyy H:mm:ss"),
        F.to_timestamp(s, "dd.MM.yyyy HH:mm:ss"),
        F.to_timestamp(s, "yyyy-MM-dd HH:mm:ss.SSSSSS"),
        F.to_timestamp(s, "yyyy-MM-dd HH:mm:ss"),
        F.to_timestamp(s, "yyyy-MM-dd"),
    )


def load_current(table: str, columns: List[str]) -> DataFrame:
    """Читает актуальные строки и проецирует только нужные поля."""
    df = spark.table(f"{SRC_DB}.{table}")
    existing = set(df.columns)
    missing = sorted(set(columns) - existing)
    if missing:
        raise ValueError(f"{table}: отсутствуют ожидаемые поля: {missing}")

    if "ctl_action" in existing:
        df = df.filter(
            F.col("ctl_action").isNull()
            | (F.upper(F.col("ctl_action").cast("string")) != "D")
        )
    if "start_dt" in existing:
        df = df.filter(_ts("start_dt").isNull() | (F.to_date(_ts("start_dt")) <= F.current_date()))
    if "end_dt" in existing:
        df = df.filter(_ts("end_dt").isNull() | (F.to_date(_ts("end_dt")) >= F.current_date()))

    exprs = []
    for c in columns:
        if c.endswith("_dttm") or c in {"start_dt", "end_dt"}:
            exprs.append(_ts(c).alias(c))
        else:
            exprs.append(F.col(c))
    return df.select(*exprs)


VIEWS = {
    "model": ("t_model", [
        "model_sid", "model_type_name", "model_ml_task_name", "model_rsk_flag",
        "model_rsk_type_name", "model_rsk_sgmnt_name", "model_stts_name",
        "model_dev_dprtmt_name",
    ]),
    "mv": ("t_model_ver", [
        "model_ver_sid", "model_sid", "model_ver_stts_name",
        "model_ver_signfcnt_ctgry_code", "model_ver_owner_dprtmt_name",
        "model_ver_dev_dprtmt_name", "model_ver_method_name", "model_ver_tgt_txt",
        "model_ver_data_type_name", "model_ver_repstry_link_txt",
        "model_ver_repstry_commit_sid", "model_ver_dev_report_sid",
        "model_ver_data_mart_link_txt", "model_ver_dev_start_plan_dttm",
        "model_ver_dev_start_fact_dttm", "model_ver_dev_end_plan_dttm",
        "model_ver_dev_end_fact_dttm", "model_ver_prevalid_flag",
        "model_ver_prevalid_rslt_name", "model_ver_sota_flag",
        "model_ver_proj_feature_flag", "model_ver_proj_feature_sid",
        "model_ver_dev_src_name", "model_ver_llm_flag", "model_ver_llm_descr_txt",
        "model_ver_montrg_auto_flag", "model_ver_montrg_auto_exclude_reason_array",
        "model_ver_crtn_dttm",
    ]),
    "sample": ("t_sample_data", [
        "sample_data_sid", "model_ver_sid", "valid_sid", "montrg_auto_rslt_sid",
        "montrg_manual_rslt_sid", "sample_data_prop_array", "sample_data_type_name",
        "sample_data_stts_name", "sample_data_not_metric_calc_flag",
    ]),
    "metric": ("t_metric", [
        "metric_sid", "sample_data_sid", "metric_name", "metric_val", "metric_stts_name",
    ]),
    "valid": ("t_valid", [
        "valid_sid", "model_ver_sid", "valid_crtn_dttm", "valid_start_fact_dttm",
        "valid_end_fact_dttm", "valid_freq_start_plan_dttm", "valid_stts_name",
        "valid_rslt_name", "valid_dprtmt_name", "valid_type_name",
        "valid_dev_return_reason_txt", "valid_not_sample_data_flag",
        "valid_src_not_table_flag", "valid_report_sid", "valid_prblm_name",
        "valid_alt_flag", "valid_alt_metric_name", "valid_alt_metric_val",
        "valid_red_zone_owner_aprvl_rsk_flag", "valid_red_zone_comt_rsk_decsn_sid",
        "valid_red_zone_comt_rsk_decsn_link_txt", "valid_agent_usg_flag",
        "valid_agent_test_quality_corr_pct", "valid_agent_test_quality_reuse_pct",
        "valid_agent_test_quantity_reuse_pct", "valid_agent_non_usg_reason_name",
        "valid_prevalid_reuse_lvl_name", "valid_prevalid_reuse_lvl_reason_array",
    ]),
    "x_auto": ("t_model_ver_x_montrg_auto", ["model_ver_sid", "montrg_auto_sid"]),
    "auto": ("t_montrg_auto", [
        "montrg_auto_sid", "montrg_auto_stts_name", "montrg_auto_next_dttm",
        "montrg_auto_proj_sched_stts_name", "montrg_auto_proj_sched_cron_txt",
        "montrg_auto_freq_name", "montrg_auto_dprtmt_name",
    ]),
    "auto_r": ("t_montrg_auto_rslt", [
        "montrg_auto_rslt_sid", "montrg_auto_sid", "montrg_auto_rslt_name",
        "montrg_auto_rslt_crtn_dttm", "montrg_auto_rslt_end_dttm",
        "montrg_auto_rslt_stts_name", "montrg_auto_rslt_prblm_name",
        "montrg_auto_rslt_metric_val", "montrg_auto_rslt_not_metric_main_flag",
        "montrg_auto_rslt_valid_not_sample_data_flag", "montrg_auto_rslt_report_sid",
    ]),
    "x_manual": ("t_model_ver_x_montrg_manual", ["model_ver_sid", "montrg_manual_sid"]),
    "manual": ("t_montrg_manual", [
        "montrg_manual_sid", "montrg_manual_stts_name", "montrg_manual_next_dttm",
        "montrg_manual_last_rslt_name", "montrg_manual_last_rslt_dttm",
        "montrg_manual_dprtmt_name",
    ]),
    "manual_r": ("t_montrg_manual_rslt", [
        "montrg_manual_rslt_sid", "montrg_manual_sid", "montrg_manual_rslt_name",
        "montrg_manual_rslt_crtn_dttm", "montrg_manual_rslt_end_dttm",
        "montrg_manual_rslt_stts_name", "montrg_manual_rslt_prblm_name",
        "montrg_manual_rslt_metric_val", "montrg_manual_rslt_not_metric_main_flag",
        "montrg_manual_rslt_valid_not_sample_data_flag", "montrg_manual_rslt_report_sid",
    ]),
    "mvp": ("t_model_ver_prom", [
        "model_ver_prom_sid", "model_ver_sid", "model_ver_prom_stts_name",
        "model_ver_prom_valid_it_rslt_qgm_flag", "model_ver_prom_implm_flag",
    ]),
    "valid_it": ("t_valid_it", [
        "valid_it_sid", "model_ver_prom_sid", "valid_it_stts_name", "valid_it_rslt_name",
        "valid_it_prblm_name", "valid_it_src_not_prom_flag", "valid_it_report_sid",
    ]),
    "x_pilot": ("t_model_ver_prom_x_pilot_implm", ["model_ver_prom_sid", "pilot_implm_sid"]),
    "pilot": ("t_pilot_implm", [
        "pilot_implm_sid", "pilot_implm_stts_name", "pilot_implm_start_plan_dttm",
        "pilot_implm_start_fact_dttm", "pilot_implm_end_fact_dttm",
    ]),
    "x_prom": ("t_model_ver_prom_x_prom_implm", ["model_ver_prom_sid", "prom_implm_sid"]),
    "prom": ("t_prom_implm", [
        "prom_implm_sid", "prom_implm_stts_name", "prom_implm_start_plan_dttm",
        "prom_implm_start_fact_dttm", "prom_implm_end_fact_dttm",
    ]),
    "genai": ("t_genai", [
        "genai_sid", "genai_stts_name", "genai_agent_flag", "genai_priority_lvl_code",
        "genai_maturity_lvl_ord", "genai_owner_dprtmt_name",
    ]),
    "gv": ("t_genai_ver", [
        "genai_ver_sid", "genai_sid", "genai_ver_stts_name",
        "genai_ver_signfcnt_ctgry_code", "genai_ver_repstry_link_txt",
        "genai_ver_repstry_commit_link_txt", "genai_ver_release_link_txt",
        "genai_ver_dev_report_sid", "genai_ver_sample_data_knowledge_lvl_name",
        "genai_ver_montrg_auto_data_flag", "genai_ver_montrg_auto_model_flag",
        "genai_ver_montrg_auto_exclude_reason_data_array",
        "genai_ver_montrg_auto_exclude_reason_model_array",
        "genai_ver_montrg_last_rslt_name", "genai_ver_montrg_last_rslt_dttm",
        "genai_ver_metric_key_code", "genai_ver_metric_key_val",
        "genai_ver_metric_assmnt_auto_cmnt_txt", "genai_ver_usg_days_qty",
        "genai_ver_usg_same_time_cnt", "genai_ver_fin_effect_plan_million_qty",
        "genai_ver_fin_effect_fact_million_qty",
    ]),
    "g_sample": ("t_genai_ver_sample_data", [
        "genai_ver_sample_data_sid", "genai_ver_sid", "genai_ver_sample_data_type_name",
        "genai_ver_sample_data_kind_name",
    ]),
    "g_valid": ("t_genai_ver_valid", [
        "genai_ver_valid_sid", "genai_ver_sid", "genai_ver_valid_stts_name",
        "genai_ver_valid_rslt_name", "genai_ver_valid_crtn_dttm",
        "genai_ver_valid_start_fact_dttm", "genai_ver_valid_end_fact_dttm",
        "genai_ver_valid_report_sid", "genai_ver_valid_alt_flag",
        "genai_ver_valid_alt_metric_name", "genai_ver_valid_alt_metric_val",
        "genai_ver_valid_metric_chg_stat_signfcnt_flag",
    ]),
}


cached_frames: List[DataFrame] = []
for view_name, (table_name, columns) in VIEWS.items():
    frame = load_current(table_name, columns).persist(StorageLevel.MEMORY_AND_DISK)
    frame.createOrReplaceTempView(view_name)
    cached_frames.append(frame)

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW mv_scope AS
SELECT * FROM mv
WHERE UPPER(COALESCE(CAST(model_ver_stts_name AS STRING), ''))
      NOT RLIKE '{CLOSED_STATUS_RE}'
""")


results: List[DataFrame] = []


def add_query(
    query_id: int,
    query_name: str,
    risk_domain: str,
    entity_type: str,
    metric_name: str,
    dim_1_name: str,
    dim_2_name: str,
    source_tables: str,
    interpretation: str,
    sql_body: str,
) -> None:
    """sql_body возвращает d1, d2, metric_value, numerator, denominator."""
    raw = spark.sql(sql_body)
    out = raw.select(
        F.lit(query_id).cast("int").alias("query_id"),
        F.lit(query_name).alias("query_name"),
        F.lit(risk_domain).alias("risk_domain"),
        F.lit(entity_type).alias("entity_type"),
        F.lit(metric_name).alias("metric_name"),
        F.lit(dim_1_name).alias("dim_1_name"),
        F.coalesce(F.col("d1").cast("string"), F.lit("$NULL$")).alias("dim_1_value"),
        F.lit(dim_2_name).alias("dim_2_name"),
        F.coalesce(F.col("d2").cast("string"), F.lit("$ALL$")).alias("dim_2_value"),
        F.col("metric_value").cast("double").alias("metric_value"),
        F.col("numerator").cast("long").alias("numerator"),
        F.col("denominator").cast("long").alias("denominator"),
        F.current_date().alias("as_of_dt"),
        F.lit(source_tables).alias("source_tables"),
        F.lit(interpretation).alias("interpretation"),
    )
    results.append(out)


def dist(qid, name, domain, entity, metric, d1_name, d2_name, sources, note, from_sql, d1, d2="NULL"):
    add_query(qid, name, domain, entity, metric, d1_name, d2_name, sources, note, f"""
        SELECT CAST({d1} AS STRING) d1, CAST({d2} AS STRING) d2,
               CAST(COUNT(*) AS DOUBLE) metric_value,
               COUNT(*) numerator, CAST(NULL AS BIGINT) denominator
        FROM {from_sql}
        GROUP BY {d1}, {d2}
    """)


def gap(qid, name, domain, entity, d1_name, d2_name, sources, note, from_sql, condition, d1, d2="NULL"):
    add_query(qid, name, domain, entity, "risk_rate_pct", d1_name, d2_name, sources, note, f"""
        SELECT CAST({d1} AS STRING) d1, CAST({d2} AS STRING) d2,
               100.0 * SUM(CASE WHEN {condition} THEN 1 ELSE 0 END) / COUNT(*) metric_value,
               SUM(CASE WHEN {condition} THEN 1 ELSE 0 END) numerator,
               COUNT(*) denominator
        FROM {from_sql}
        GROUP BY {d1}, {d2}
    """)


sig = "COALESCE(model_ver_signfcnt_ctgry_code, '$NULL$')"
gsig = "COALESCE(genai_ver_signfcnt_ctgry_code, '$NULL$')"
truth = lambda c: f"COALESCE(LOWER(CAST({c} AS STRING)) IN ('true','1','да','yes','y'), FALSE)"
falsehood = lambda c: f"COALESCE(LOWER(TRIM(CAST({c} AS STRING))) IN ('false','0','нет','no','n'), FALSE)"


def blank(c):
    """NULL, пустая строка и строковые технические NULL."""
    return (
        f"({c} IS NULL OR "
        f"LOWER(TRIM(CAST({c} AS STRING))) IN ('', 'null', 'none'))"
    )


def artifact_missing(c):
    """Отсутствующий идентификатор/артефакт с учётом фактических заглушек."""
    return (
        f"({c} IS NULL OR LOWER(TRIM(CAST({c} AS STRING))) "
        f"IN ('', 'null', 'none', '-', 'нет', 'n/a', 'na'))"
    )


def real_date(c):
    """Исключает выявленные в профиле даты-заглушки 1901/2099/2999."""
    return (
        f"COALESCE((DATE({c}) >= DATE '2000-01-01' "
        f"AND DATE({c}) < DATE '2090-01-01'), FALSE)"
    )


# 1–15. Портфель, документация, данные и разработка
dist(1, "Портфель версий по значимости и статусу", "portfolio", "MODEL", "version_count",
     "significance", "status", "t_model_ver", "Базовая структура активного портфеля.",
     "mv", sig, "COALESCE(model_ver_stts_name, '$NULL$')")

dist(2, "Модели по типу и ML-задаче", "portfolio", "MODEL", "version_count",
     "model_type", "ml_task", "t_model,t_model_ver", "Концентрация типов моделей и задач.",
     "mv_scope v JOIN model m ON v.model_sid=m.model_sid",
     "COALESCE(m.model_type_name, '$NULL$')", "COALESCE(m.model_ml_task_name, '$NULL$')")

dist(3, "Риск-модели по виду и сегменту риска", "portfolio", "MODEL", "version_count",
     "risk_type", "risk_segment", "t_model,t_model_ver", "Профиль версий риск-моделей.",
     f"mv_scope v JOIN model m ON v.model_sid=m.model_sid WHERE {truth('m.model_rsk_flag')}",
     "COALESCE(m.model_rsk_type_name, '$NULL$')", "COALESCE(m.model_rsk_sgmnt_name, '$NULL$')")

dist(4, "Критичные версии по подразделению-владельцу", "portfolio", "MODEL", "version_count",
     "owner_department", "significance", "t_model_ver", "Концентрация критичных A/B-версий по владельцам.",
     f"mv_scope WHERE model_ver_signfcnt_ctgry_code IN {CRITICAL_SIGNIFICANCE}",
     "COALESCE(model_ver_owner_dprtmt_name, '$NULL$')", sig)

add_query(5, "Пробелы ключевых метаданных", "documentation", "MODEL", "risk_rate_pct",
          "missing_field", "significance", "t_model_ver", "Доля активных версий с незаполненным ключевым полем.", f"""
    SELECT indicator d1, significance d2,
           100.0*SUM(is_gap)/COUNT(*) metric_value, SUM(is_gap) numerator, COUNT(*) denominator
    FROM (
      SELECT {sig} significance,
             STACK(4,
               'owner_department', CASE WHEN {blank('model_ver_owner_dprtmt_name')} THEN 1 ELSE 0 END,
               'method', CASE WHEN {blank('model_ver_method_name')} THEN 1 ELSE 0 END,
               'target', CASE WHEN {blank('model_ver_tgt_txt')} THEN 1 ELSE 0 END,
               'significance', CASE WHEN {blank('model_ver_signfcnt_ctgry_code')} THEN 1 ELSE 0 END
             ) AS (indicator, is_gap)
      FROM mv_scope
    ) s GROUP BY indicator, significance
""")

gap(6, "Отсутствие ссылки на репозиторий", "reproducibility", "MODEL", "significance", "dev_source",
    "t_model_ver", "Без ссылки нельзя воспроизвести код версии.", "mv_scope",
    artifact_missing("model_ver_repstry_link_txt"),
    sig, "COALESCE(model_ver_dev_src_name, '$NULL$')")

gap(7, "Отсутствие идентификатора коммита", "reproducibility", "MODEL", "significance", "dev_source",
    "t_model_ver", "Без commit ID код версии не фиксирован однозначно.", "mv_scope",
    artifact_missing("model_ver_repstry_commit_sid"),
    sig, "COALESCE(model_ver_dev_src_name, '$NULL$')")

gap(8, "Отсутствие отчёта о разработке", "documentation", "MODEL", "significance", "status",
    "t_model_ver", "Проверяет наличие доказательной базы разработки.", "mv_scope",
    artifact_missing("model_ver_dev_report_sid"),
    sig, "COALESCE(model_ver_stts_name, '$NULL$')")

add_query(9, "Отсутствие target или типа данных", "data_governance", "MODEL", "risk_rate_pct",
          "missing_field", "significance", "t_model_ver", "Неполная постановка задачи или описание входных данных.", f"""
    SELECT indicator d1, significance d2, 100.0*SUM(is_gap)/COUNT(*) metric_value,
           SUM(is_gap) numerator, COUNT(*) denominator
    FROM (
      SELECT {sig} significance,
             STACK(2,
               'target', CASE WHEN {blank('model_ver_tgt_txt')} THEN 1 ELSE 0 END,
               'data_type', CASE WHEN {blank('model_ver_data_type_name')} THEN 1 ELSE 0 END
             ) AS (indicator, is_gap)
      FROM mv_scope
    ) s GROUP BY indicator, significance
""")

gap(10, "Версии без связанных выборок", "data_governance", "MODEL", "significance", "status",
    "t_model_ver,t_sample_data", "Разрыв связи между версией модели и данными.",
    "mv_scope v LEFT JOIN (SELECT DISTINCT model_ver_sid FROM sample WHERE model_ver_sid IS NOT NULL) s ON v.model_ver_sid=s.model_ver_sid",
    "s.model_ver_sid IS NULL", sig, "COALESCE(v.model_ver_stts_name, '$NULL$')")

gap(11, "Версии без OOT-выборки", "data_governance", "MODEL", "significance", "status",
    "t_model_ver,t_sample_data", "Отсутствие OOT снижает доказанность временной устойчивости.", f"""
      mv_scope v LEFT JOIN (
        SELECT DISTINCT model_ver_sid FROM sample
        WHERE UPPER(COALESCE(CAST(sample_data_prop_array AS STRING),'')) RLIKE '{OOT_RE}'
      ) s ON v.model_ver_sid=s.model_ver_sid
    """, "s.model_ver_sid IS NULL", sig, "COALESCE(v.model_ver_stts_name, '$NULL$')")

add_query(12, "Выборки без пригодных метрик", "data_governance", "MODEL", "risk_rate_pct",
          "sample_type", "gap_type", "t_sample_data,t_metric", "Выявляет выборки, для которых количественная оценка отсутствует или запрещена.", f"""
    WITH x AS (
      SELECT s.sample_data_sid, COALESCE(s.sample_data_type_name,'$NULL$') sample_type,
             MAX(CASE WHEN m.metric_sid IS NOT NULL AND NOT ({blank('m.metric_val')}) THEN 1 ELSE 0 END) has_metric,
             MAX(CASE WHEN {truth('s.sample_data_not_metric_calc_flag')} THEN 1 ELSE 0 END) no_calc
      FROM sample s LEFT JOIN metric m ON s.sample_data_sid=m.sample_data_sid
      GROUP BY s.sample_data_sid, COALESCE(s.sample_data_type_name,'$NULL$')
    ), y AS (
      SELECT sample_type, STACK(2, 'no_metric', 1-has_metric, 'metric_calc_disabled', no_calc) AS (gap_type,is_gap) FROM x
    )
    SELECT sample_type d1, gap_type d2, 100.0*SUM(is_gap)/COUNT(*) metric_value,
           SUM(is_gap) numerator, COUNT(*) denominator FROM y GROUP BY sample_type,gap_type
""")

gap(13, "Просроченная разработка", "development", "MODEL", "department", "significance",
    "t_model_ver", "Факт окончания позже плана либо открытая работа после плановой даты.", "mv_scope",
    f"{real_date('model_ver_dev_end_plan_dttm')} AND "
    f"((NOT {real_date('model_ver_dev_end_fact_dttm')} AND DATE(model_ver_dev_end_plan_dttm)<CURRENT_DATE()) "
    f"OR ({real_date('model_ver_dev_end_fact_dttm')} AND model_ver_dev_end_fact_dttm>model_ver_dev_end_plan_dttm))",
    "COALESCE(model_ver_dev_dprtmt_name,'$NULL$')", sig)

add_query(14, "Длительность разработки", "development", "MODEL", "duration_days",
          "department", "statistic", "t_model_ver", "Медиана и p90 длительности завершённой разработки.", f"""
    WITH x AS (
      SELECT COALESCE(model_ver_dev_dprtmt_name,'$NULL$') department,
             DATEDIFF(DATE(model_ver_dev_end_fact_dttm),DATE(model_ver_dev_start_fact_dttm)) days
      FROM mv WHERE {real_date('model_ver_dev_start_fact_dttm')}
                    AND {real_date('model_ver_dev_end_fact_dttm')}
    ), stats AS (
      SELECT department,PERCENTILE_APPROX(days,0.50) p50,
             PERCENTILE_APPROX(days,0.90) p90,COUNT(*) n
      FROM x WHERE days>=0 GROUP BY department
    )
    SELECT department d1,'p50' d2,CAST(p50 AS DOUBLE) metric_value,
           CAST(NULL AS BIGINT) numerator,n denominator FROM stats
    UNION ALL
    SELECT department,'p90',CAST(p90 AS DOUBLE),CAST(NULL AS BIGINT),n FROM stats
""")

gap(15, "Неконсистентность FeatureStore", "data_governance", "MODEL", "significance", "status",
    "t_model_ver", "Флаг FeatureStore установлен, но идентификатор проекта отсутствует.", "mv_scope",
    f"{truth('model_ver_proj_feature_flag')} AND {blank('model_ver_proj_feature_sid')}",
    sig, "COALESCE(model_ver_stts_name,'$NULL$')")


# 16–32. Валидация
gap(16, "Покрытие валидацией", "validation", "MODEL", "significance", "status",
    "t_model_ver,t_valid", "Доля активных версий без единой валидации.",
    "mv_scope v LEFT JOIN (SELECT DISTINCT model_ver_sid FROM valid) x ON v.model_ver_sid=x.model_ver_sid",
    "x.model_ver_sid IS NULL", sig, "COALESCE(v.model_ver_stts_name,'$NULL$')")

dist(17, "Результаты валидации", "validation", "MODEL", "validation_count",
     "result", "significance", "t_valid,t_model_ver", "Распределение итогов валидации.",
     "valid v LEFT JOIN mv m ON v.model_ver_sid=m.model_ver_sid",
     "COALESCE(v.valid_rslt_name,'$NULL$')", "COALESCE(m.model_ver_signfcnt_ctgry_code,'$NULL$')")

gap(18, "Возраст незавершённой валидации", "validation", "MODEL", "department", "status",
    "t_valid", f"Доля открытых валидаций старше {OPEN_SLA_DAYS} дней.",
    f"valid WHERE UPPER(COALESCE(valid_stts_name,'')) NOT RLIKE '{CLOSED_STATUS_RE}'",
    f"{real_date('COALESCE(valid_start_fact_dttm,valid_crtn_dttm)')} AND "
    f"DATEDIFF(CURRENT_DATE(),DATE(COALESCE(valid_start_fact_dttm,valid_crtn_dttm)))>{OPEN_SLA_DAYS}",
    "COALESCE(valid_dprtmt_name,'$NULL$')", "COALESCE(valid_stts_name,'$NULL$')")

gap(19, "Просроченная периодическая валидация", "validation", "MODEL", "department", "status",
    "t_valid", "Плановая дата прошла, а валидация не закрыта.", "valid",
    f"{real_date('valid_freq_start_plan_dttm')} AND DATE(valid_freq_start_plan_dttm)<CURRENT_DATE() "
    f"AND UPPER(COALESCE(valid_stts_name,'')) NOT RLIKE '{CLOSED_STATUS_RE}'",
    "COALESCE(valid_dprtmt_name,'$NULL$')", "COALESCE(valid_stts_name,'$NULL$')")

add_query(20, "Длительность валидации", "validation", "MODEL", "duration_days",
          "department", "statistic", "t_valid", "Медиана и p90 длительности завершённой валидации.", f"""
    WITH x AS (
      SELECT COALESCE(valid_dprtmt_name,'$NULL$') department,
             DATEDIFF(DATE(valid_end_fact_dttm),DATE(valid_start_fact_dttm)) days
      FROM valid WHERE {real_date('valid_start_fact_dttm')}
                       AND {real_date('valid_end_fact_dttm')}
    ), stats AS (
      SELECT department,PERCENTILE_APPROX(days,0.50) p50,
             PERCENTILE_APPROX(days,0.90) p90,COUNT(*) n
      FROM x WHERE days>=0 GROUP BY department
    )
    SELECT department d1,'p50' d2,CAST(p50 AS DOUBLE) metric_value,
           CAST(NULL AS BIGINT) numerator,n denominator FROM stats
    UNION ALL
    SELECT department,'p90',CAST(p90 AS DOUBLE),CAST(NULL AS BIGINT),n FROM stats
""")

dist(21, "Возвраты разработчику", "validation", "MODEL", "return_count",
     "return_reason", "department", "t_valid", "Частые причины возврата показывают системные дефекты разработки.",
     "valid WHERE valid_dev_return_reason_txt IS NOT NULL AND TRIM(CAST(valid_dev_return_reason_txt AS STRING))<>''",
     "valid_dev_return_reason_txt", "COALESCE(valid_dprtmt_name,'$NULL$')")

gap(22, "Валидации без выборки", "validation", "MODEL", "result", "type",
    "t_valid", "Доля валидаций, помеченных как проведённые без выборки.", "valid",
    truth("valid_not_sample_data_flag"), "COALESCE(valid_rslt_name,'$NULL$')", "COALESCE(valid_type_name,'$NULL$')")

gap(23, "Нетабличные источники валидации", "validation", "MODEL", "result", "type",
    "t_valid", "Нетабличные данные ограничивают воспроизводимость и автоматизацию.", "valid",
    truth("valid_src_not_table_flag"), "COALESCE(valid_rslt_name,'$NULL$')", "COALESCE(valid_type_name,'$NULL$')")

gap(24, "Валидации без отчёта", "validation", "MODEL", "result", "department",
    "t_valid", "Результат не подкреплён ссылкой на отчёт.", "valid",
    artifact_missing("valid_report_sid"),
    "COALESCE(valid_rslt_name,'$NULL$')", "COALESCE(valid_dprtmt_name,'$NULL$')")

gap(25, "Красный результат без принятия риска владельцем", "validation", "MODEL", "department", "result",
    "t_valid", "Красная модель требует явного принятия риска владельцем.",
    f"valid WHERE UPPER(COALESCE(valid_rslt_name,'')) RLIKE '{RED_RESULT_RE}'",
    f"NOT ({truth('valid_red_zone_owner_aprvl_rsk_flag')})",
    "COALESCE(valid_dprtmt_name,'$NULL$')", "COALESCE(valid_rslt_name,'$NULL$')")

gap(26, "Красный результат без решения КРГ", "validation", "MODEL", "department", "result",
    "t_valid", "Проверяет наличие ID и ссылки решения КРГ.",
    f"valid WHERE UPPER(COALESCE(valid_rslt_name,'')) RLIKE '{RED_RESULT_RE}'",
    f"{artifact_missing('valid_red_zone_comt_rsk_decsn_sid')} OR "
    f"{artifact_missing('valid_red_zone_comt_rsk_decsn_link_txt')}",
    "COALESCE(valid_dprtmt_name,'$NULL$')", "COALESCE(valid_rslt_name,'$NULL$')")

dist(27, "Типы ошибок валидации", "validation", "MODEL", "problem_count",
     "problem", "result", "t_valid", "Показывает наиболее частые классы выявленных рисков.",
     "valid WHERE valid_prblm_name IS NOT NULL", "valid_prblm_name", "COALESCE(valid_rslt_name,'$NULL$')")

add_query(28, "Альтернативное моделирование и метрики", "validation", "MODEL", "risk_rate_pct",
          "indicator", "significance", "t_valid,t_model_ver", "Покрытие независимой альтернативой и полнота её метрики.", f"""
    WITH x AS (
      SELECT COALESCE(m.model_ver_signfcnt_ctgry_code,'$NULL$') significance,
             STACK(2,
               'alt_not_run', CASE WHEN NOT ({truth('v.valid_alt_flag')}) THEN 1 ELSE 0 END,
               'alt_metric_missing', CASE WHEN {truth('v.valid_alt_flag')} AND ({blank('v.valid_alt_metric_name')} OR {blank('v.valid_alt_metric_val')}) THEN 1 ELSE 0 END
             ) AS (indicator,is_gap)
      FROM valid v LEFT JOIN mv m ON v.model_ver_sid=m.model_ver_sid
    ) SELECT indicator d1, significance d2, 100.0*SUM(is_gap)/COUNT(*) metric_value,
             SUM(is_gap) numerator, COUNT(*) denominator FROM x GROUP BY indicator,significance
""")

gap(29, "Заполненность признака агента-валидатора", "agent_validation", "MODEL", "type", "result",
    "t_valid", "Пустой признак не считается отказом от агента: это отдельный риск качества данных.", "valid",
    blank("valid_agent_usg_flag"), "COALESCE(valid_type_name,'$NULL$')", "COALESCE(valid_rslt_name,'$NULL$')")

add_query(30, "Готовность данных агентной валидации", "agent_validation", "MODEL", "risk_rate_pct",
          "missing_field", "validation_type", "t_valid", "Заполненность показателей качества и причин неиспользования агента.", f"""
    SELECT indicator d1, validation_type d2,
           100.0*SUM(is_gap)/COUNT(*) metric_value,
           SUM(is_gap) numerator, COUNT(*) denominator
    FROM (
      SELECT COALESCE(valid_type_name,'$NULL$') validation_type,
             STACK(4,
               'quality_correct_pct', CASE WHEN valid_agent_test_quality_corr_pct IS NULL THEN 1 ELSE 0 END,
               'quality_reuse_pct', CASE WHEN valid_agent_test_quality_reuse_pct IS NULL THEN 1 ELSE 0 END,
               'quantity_reuse_pct', CASE WHEN valid_agent_test_quantity_reuse_pct IS NULL THEN 1 ELSE 0 END,
               'non_usage_reason', CASE WHEN {blank('valid_agent_non_usg_reason_name')} THEN 1 ELSE 0 END
             ) AS (indicator,is_gap)
      FROM valid
    ) x GROUP BY indicator,validation_type
""")

dist(31, "Ручная и автоматическая валидация", "agent_validation", "MODEL", "validation_count",
     "validation_type", "status", "t_valid", "Фактический уровень автоматизации процесса валидации.",
     "valid", "COALESCE(valid_type_name,'$NULL$')", "COALESCE(valid_stts_name,'$NULL$')")

dist(32, "Переиспользование превалидации", "validation", "MODEL", "validation_count",
     "reuse_level", "reason", "t_valid", "Показывает степень и причины неполного переиспользования превалидации.",
     "valid", "COALESCE(valid_prevalid_reuse_lvl_name,'$NULL$')", "COALESCE(CAST(valid_prevalid_reuse_lvl_reason_array AS STRING),'$NULL$')")


# 33–42. Мониторинг, внедрение и целостность
add_query(33, "Связь версии с автоматическим мониторингом", "monitoring", "MODEL", "risk_rate_pct",
          "indicator", "significance", "t_model_ver,t_model_ver_x_montrg_auto,t_montrg_auto",
          "Раздельно показывает пустой признак мониторинга и отсутствие связи; таблица связей в профиле пуста.", f"""
    WITH links AS (
      SELECT model_ver_sid,1 has_link FROM x_auto GROUP BY model_ver_sid
    ), x AS (
      SELECT {sig} significance,
             STACK(2,
               'monitoring_flag_missing', CASE WHEN {blank('v.model_ver_montrg_auto_flag')} THEN 1 ELSE 0 END,
               'monitoring_link_missing', CASE WHEN l.model_ver_sid IS NULL THEN 1 ELSE 0 END
             ) AS (indicator,is_gap)
      FROM mv_scope v LEFT JOIN links l ON v.model_ver_sid=l.model_ver_sid
    )
    SELECT indicator d1,significance d2,100.0*SUM(is_gap)/COUNT(*) metric_value,
           SUM(is_gap) numerator,COUNT(*) denominator
    FROM x GROUP BY indicator,significance
""")

gap(34, "Просроченный автоматический мониторинг", "monitoring", "MODEL", "department", "status",
    "t_montrg_auto", "Дата следующего запуска прошла, карточка не закрыта.", "auto",
    f"{real_date('montrg_auto_next_dttm')} AND DATE(montrg_auto_next_dttm)<CURRENT_DATE() "
    f"AND UPPER(COALESCE(montrg_auto_stts_name,'')) NOT RLIKE '{CLOSED_STATUS_RE}'",
    "COALESCE(montrg_auto_dprtmt_name,'$NULL$')", "COALESCE(montrg_auto_stts_name,'$NULL$')")

add_query(35, "Разрывы расписания автомониторинга", "monitoring", "MODEL", "risk_rate_pct",
          "schedule_gap", "schedule_status", "t_montrg_auto", "Выявляет неактивное расписание и отсутствующие cron/дату запуска.", f"""
    SELECT indicator d1, schedule_status d2, 100.0*SUM(is_gap)/COUNT(*) metric_value,
           SUM(is_gap) numerator, COUNT(*) denominator
    FROM (
      SELECT COALESCE(montrg_auto_proj_sched_stts_name,'$NULL$') schedule_status,
             STACK(3,
               'inactive_schedule', CASE WHEN UPPER(COALESCE(montrg_auto_proj_sched_stts_name,'')) RLIKE 'INACTIVE|DISABLED|ОТКЛ' THEN 1 ELSE 0 END,
               'cron_missing', CASE WHEN {blank('montrg_auto_proj_sched_cron_txt')} THEN 1 ELSE 0 END,
               'next_run_missing', CASE WHEN NOT {real_date('montrg_auto_next_dttm')} THEN 1 ELSE 0 END
             ) AS (indicator,is_gap)
      FROM auto
    ) x GROUP BY indicator,schedule_status
""")

add_query(36, "Результаты и пробелы автомониторинга", "monitoring", "MODEL", "result_count_or_gap_pct",
          "indicator_or_result", "result", "t_montrg_auto_rslt,t_sample_data", "Цвет результата и отсутствие метрики, выборки или отчёта.", f"""
    WITH x AS (
      SELECT r.*, MAX(CASE WHEN s.sample_data_sid IS NOT NULL THEN 1 ELSE 0 END) OVER(PARTITION BY r.montrg_auto_rslt_sid) has_sample
      FROM auto_r r LEFT JOIN sample s ON r.montrg_auto_rslt_sid=s.montrg_auto_rslt_sid
    ), one_row AS (SELECT DISTINCT * FROM x), gaps AS (
      SELECT COALESCE(montrg_auto_rslt_name,'$NULL$') result,
             STACK(3,
               'metric_missing', CASE WHEN montrg_auto_rslt_metric_val IS NULL OR {truth('montrg_auto_rslt_not_metric_main_flag')} THEN 1 ELSE 0 END,
               'sample_missing', CASE WHEN has_sample=0 OR {truth('montrg_auto_rslt_valid_not_sample_data_flag')} THEN 1 ELSE 0 END,
               'report_missing', CASE WHEN montrg_auto_rslt_report_sid IS NULL THEN 1 ELSE 0 END
             ) AS (indicator,is_gap)
      FROM one_row
    ) SELECT indicator d1,result d2,100.0*SUM(is_gap)/COUNT(*) metric_value,
             SUM(is_gap) numerator,COUNT(*) denominator FROM gaps GROUP BY indicator,result
""")

add_query(37, "Актуальность ручного мониторинга", "monitoring", "MODEL", "risk_rate_pct",
          "indicator", "status", "t_montrg_manual",
          "Оценивает карточки напрямую: таблица связи с версиями в профиле пуста.", f"""
    WITH x AS (
      SELECT COALESCE(montrg_manual_stts_name,'$NULL$') status,
             STACK(3,
               'last_result_missing', CASE WHEN {blank('montrg_manual_last_rslt_name')} THEN 1 ELSE 0 END,
               'last_result_missing_or_stale', CASE WHEN NOT {real_date('montrg_manual_last_rslt_dttm')}
                 OR DATEDIFF(CURRENT_DATE(),DATE(montrg_manual_last_rslt_dttm))>{STALE_MONITORING_DAYS} THEN 1 ELSE 0 END,
               'next_run_overdue', CASE WHEN {real_date('montrg_manual_next_dttm')}
                 AND DATE(montrg_manual_next_dttm)<CURRENT_DATE()
                 AND UPPER(COALESCE(montrg_manual_stts_name,'')) NOT RLIKE '{CLOSED_STATUS_RE}' THEN 1 ELSE 0 END
             ) AS (indicator,is_gap)
      FROM manual
    ) SELECT indicator d1,status d2,100.0*SUM(is_gap)/COUNT(*) metric_value,
             SUM(is_gap) numerator,COUNT(*) denominator FROM x GROUP BY indicator,status
""")

add_query(38, "Результаты и пробелы ручного мониторинга", "monitoring", "MODEL", "result_count_or_gap_pct",
          "indicator", "result", "t_montrg_manual_rslt,t_sample_data", "Цвет результата и отсутствие метрики, выборки или отчёта.", f"""
    WITH x AS (
      SELECT r.*, MAX(CASE WHEN s.sample_data_sid IS NOT NULL THEN 1 ELSE 0 END) OVER(PARTITION BY r.montrg_manual_rslt_sid) has_sample
      FROM manual_r r LEFT JOIN sample s ON r.montrg_manual_rslt_sid=s.montrg_manual_rslt_sid
    ), one_row AS (SELECT DISTINCT * FROM x), gaps AS (
      SELECT COALESCE(montrg_manual_rslt_name,'$NULL$') result,
             STACK(3,
               'metric_missing', CASE WHEN montrg_manual_rslt_metric_val IS NULL OR {truth('montrg_manual_rslt_not_metric_main_flag')} THEN 1 ELSE 0 END,
               'sample_missing', CASE WHEN has_sample=0 OR {truth('montrg_manual_rslt_valid_not_sample_data_flag')} THEN 1 ELSE 0 END,
               'report_missing', CASE WHEN montrg_manual_rslt_report_sid IS NULL THEN 1 ELSE 0 END
             ) AS (indicator,is_gap)
      FROM one_row
    ) SELECT indicator d1,result d2,100.0*SUM(is_gap)/COUNT(*) metric_value,
             SUM(is_gap) numerator,COUNT(*) denominator FROM gaps GROUP BY indicator,result
""")

add_query(39, "Проблемы мониторинга по каналу", "monitoring", "MODEL", "problem_count",
          "channel", "problem", "t_montrg_auto_rslt,t_montrg_manual_rslt",
          "Сравнение причин риска без недостоверного соединения через пустые таблицы связей.", f"""
    WITH issues AS (
      SELECT 'automatic' channel,montrg_auto_rslt_prblm_name problem
      FROM auto_r WHERE NOT ({blank('montrg_auto_rslt_prblm_name')})
      UNION ALL
      SELECT 'manual',montrg_manual_rslt_prblm_name
      FROM manual_r WHERE NOT ({blank('montrg_manual_rslt_prblm_name')})
    )
    SELECT channel d1,problem d2,CAST(COUNT(*) AS DOUBLE) metric_value,
           COUNT(*) numerator,CAST(NULL AS BIGINT) denominator
    FROM issues GROUP BY channel,problem
""")

add_query(40, "ИТ-валидация и QGM промышленной версии", "implementation", "MODEL", "risk_rate_pct",
          "indicator", "prom_status", "t_model_ver_prom,t_valid_it", "Пустой QGM-флаг учитывается как отсутствие данных, а не как отрицательное решение.", f"""
    WITH x AS (
      SELECT COALESCE(p.model_ver_prom_stts_name,'$NULL$') prom_status,
             STACK(3,
               'it_validation_missing', CASE WHEN i.model_ver_prom_sid IS NULL THEN 1 ELSE 0 END,
               'qgm_flag_missing', CASE WHEN {blank('p.model_ver_prom_valid_it_rslt_qgm_flag')} THEN 1 ELSE 0 END,
               'implementation_flag_missing', CASE WHEN {blank('p.model_ver_prom_implm_flag')} THEN 1 ELSE 0 END
             ) AS (indicator,is_gap)
      FROM mvp p LEFT JOIN (SELECT DISTINCT model_ver_prom_sid FROM valid_it) i
        ON p.model_ver_prom_sid=i.model_ver_prom_sid
    ) SELECT indicator d1,prom_status d2,100.0*SUM(is_gap)/COUNT(*) metric_value,
             SUM(is_gap) numerator,COUNT(*) denominator FROM x GROUP BY indicator,prom_status
""")

add_query(41, "Просрочка пилота и промышленного внедрения", "implementation", "MODEL", "risk_rate_pct",
          "implementation_type", "status", "implementation tables", "Плановая дата старта прошла, но фактический старт отсутствует или опоздал; даты-заглушки исключены.", f"""
    WITH impl AS (
      SELECT 'pilot' kind, p.pilot_implm_stts_name status, p.pilot_implm_start_plan_dttm plan_dt, p.pilot_implm_start_fact_dttm fact_dt
      FROM pilot p
      UNION ALL
      SELECT 'industrial', p.prom_implm_stts_name, p.prom_implm_start_plan_dttm, p.prom_implm_start_fact_dttm FROM prom p
    ) SELECT kind d1,COALESCE(status,'$NULL$') d2,
             100.0*SUM(CASE WHEN {real_date('plan_dt')} AND
               ((NOT {real_date('fact_dt')} AND DATE(plan_dt)<CURRENT_DATE()) OR
                ({real_date('fact_dt')} AND fact_dt>plan_dt)) THEN 1 ELSE 0 END)/COUNT(*) metric_value,
             SUM(CASE WHEN {real_date('plan_dt')} AND
               ((NOT {real_date('fact_dt')} AND DATE(plan_dt)<CURRENT_DATE()) OR
                ({real_date('fact_dt')} AND fact_dt>plan_dt)) THEN 1 ELSE 0 END) numerator,
             COUNT(*) denominator FROM impl GROUP BY kind,COALESCE(status,'$NULL$')
""")

add_query(42, "Нарушения ссылочной целостности", "data_quality", "ALL", "orphan_rate_pct",
          "relation", "", "key entity and bridge tables", "Доля дочерних записей, для которых отсутствует родитель.", """
    WITH checks AS (
      SELECT 'MODEL_VER->MODEL' relation, SUM(CASE WHEN m.model_sid IS NULL THEN 1 ELSE 0 END) bad, COUNT(*) total
        FROM mv v LEFT JOIN model m ON v.model_sid=m.model_sid
      UNION ALL SELECT 'VALID->MODEL_VER',SUM(CASE WHEN v.model_ver_sid IS NULL THEN 1 ELSE 0 END),COUNT(*)
        FROM valid x LEFT JOIN mv v ON x.model_ver_sid=v.model_ver_sid
      UNION ALL SELECT 'SAMPLE->MODEL_VER',SUM(CASE WHEN v.model_ver_sid IS NULL THEN 1 ELSE 0 END),COUNT(*)
        FROM sample x LEFT JOIN mv v ON x.model_ver_sid=v.model_ver_sid WHERE x.model_ver_sid IS NOT NULL
      UNION ALL SELECT 'METRIC->SAMPLE',SUM(CASE WHEN s.sample_data_sid IS NULL THEN 1 ELSE 0 END),COUNT(*)
        FROM metric x LEFT JOIN sample s ON x.sample_data_sid=s.sample_data_sid WHERE x.sample_data_sid IS NOT NULL
      UNION ALL SELECT 'X_AUTO->AUTO',SUM(CASE WHEN a.montrg_auto_sid IS NULL THEN 1 ELSE 0 END),COUNT(*)
        FROM x_auto x LEFT JOIN auto a ON x.montrg_auto_sid=a.montrg_auto_sid
      UNION ALL SELECT 'X_MANUAL->MANUAL',SUM(CASE WHEN m.montrg_manual_sid IS NULL THEN 1 ELSE 0 END),COUNT(*)
        FROM x_manual x LEFT JOIN manual m ON x.montrg_manual_sid=m.montrg_manual_sid
      UNION ALL SELECT 'GENAI_VER->GENAI',SUM(CASE WHEN g.genai_sid IS NULL THEN 1 ELSE 0 END),COUNT(*)
        FROM gv v LEFT JOIN genai g ON v.genai_sid=g.genai_sid
      UNION ALL SELECT 'GENAI_SAMPLE->GENAI_VER',SUM(CASE WHEN v.genai_ver_sid IS NULL THEN 1 ELSE 0 END),COUNT(*)
        FROM g_sample x LEFT JOIN gv v ON x.genai_ver_sid=v.genai_ver_sid
      UNION ALL SELECT 'GENAI_VALID->GENAI_VER',SUM(CASE WHEN v.genai_ver_sid IS NULL THEN 1 ELSE 0 END),COUNT(*)
        FROM g_valid x LEFT JOIN gv v ON x.genai_ver_sid=v.genai_ver_sid
      UNION ALL SELECT 'VALID_IT->MODEL_VER_PROM',SUM(CASE WHEN p.model_ver_prom_sid IS NULL THEN 1 ELSE 0 END),COUNT(*)
        FROM valid_it x LEFT JOIN mvp p ON x.model_ver_prom_sid=p.model_ver_prom_sid
    ) SELECT relation d1,CAST(NULL AS STRING) d2,
             CASE WHEN total=0 THEN CAST(NULL AS DOUBLE) ELSE 100.0*bad/total END metric_value,
             COALESCE(bad,0) numerator,total denominator FROM checks
""")


# 43–50. GenAI и агенты
add_query(43, "Портфель GenAI и агентов", "genai_portfolio", "GENAI", "version_count",
          "portfolio_attribute", "value", "t_genai,t_genai_ver", "Статус, значимость, признак агента, критичность и зрелость портфеля.", """
    SELECT attribute d1,value d2,CAST(COUNT(*) AS DOUBLE) metric_value,
           COUNT(*) numerator,CAST(NULL AS BIGINT) denominator
    FROM (
      SELECT STACK(5,
        'version_status',COALESCE(CAST(v.genai_ver_stts_name AS STRING),'$NULL$'),
        'significance',COALESCE(CAST(v.genai_ver_signfcnt_ctgry_code AS STRING),'$NULL$'),
        'agent_flag',COALESCE(CAST(g.genai_agent_flag AS STRING),'$NULL$'),
        'priority',COALESCE(UPPER(TRIM(CAST(g.genai_priority_lvl_code AS STRING))),'$NULL$'),
        'maturity',COALESCE(CAST(g.genai_maturity_lvl_ord AS STRING),'$NULL$')
      ) AS (attribute,value)
      FROM gv v LEFT JOIN genai g ON v.genai_sid=g.genai_sid
    ) x GROUP BY attribute,value
""")

add_query(44, "Документирование и воспроизводимость GenAI", "genai_governance", "GENAI", "risk_rate_pct",
          "missing_field", "significance", "t_genai,t_genai_ver", "Пробелы владельца, репозитория, коммита, релиза и отчёта разработки.", f"""
    SELECT indicator d1,significance d2,100.0*SUM(is_gap)/COUNT(*) metric_value,SUM(is_gap) numerator,COUNT(*) denominator
    FROM (
      SELECT {gsig} significance,
             STACK(5,
               'owner',CASE WHEN {blank('g.genai_owner_dprtmt_name')} THEN 1 ELSE 0 END,
               'repository',CASE WHEN {artifact_missing('v.genai_ver_repstry_link_txt')} THEN 1 ELSE 0 END,
               'commit',CASE WHEN {artifact_missing('v.genai_ver_repstry_commit_link_txt')} THEN 1 ELSE 0 END,
               'release',CASE WHEN {artifact_missing('v.genai_ver_release_link_txt')} THEN 1 ELSE 0 END,
               'development_report',CASE WHEN {artifact_missing('v.genai_ver_dev_report_sid')} THEN 1 ELSE 0 END
             ) AS (indicator,is_gap)
      FROM gv v LEFT JOIN genai g ON v.genai_sid=g.genai_sid
    ) x GROUP BY indicator,significance
""")

add_query(45, "Данные GenAI", "genai_data", "GENAI", "risk_rate_pct",
          "indicator", "significance", "t_genai_ver,t_genai_ver_sample_data", "Отсутствие выборки или уровня знания данных.", f"""
    WITH x AS (
      SELECT v.*,CASE WHEN s.genai_ver_sid IS NULL THEN 0 ELSE 1 END has_sample
      FROM gv v LEFT JOIN (SELECT DISTINCT genai_ver_sid FROM g_sample) s ON v.genai_ver_sid=s.genai_ver_sid
    ), y AS (
      SELECT {gsig} significance,STACK(2,
        'sample_missing',CASE WHEN has_sample=0 THEN 1 ELSE 0 END,
        'data_knowledge_level_missing',CASE WHEN {blank('genai_ver_sample_data_knowledge_lvl_name')} THEN 1 ELSE 0 END
      ) AS (indicator,is_gap) FROM x
    ) SELECT indicator d1,significance d2,100.0*SUM(is_gap)/COUNT(*) metric_value,SUM(is_gap) numerator,COUNT(*) denominator
      FROM y GROUP BY indicator,significance
""")

add_query(46, "Покрытие и результаты валидации GenAI", "genai_validation", "GENAI", "version_count",
          "validation_result", "significance", "t_genai_ver,t_genai_ver_valid", "Распределение результатов, включая версии без валидации.", f"""
    WITH latest AS (
      SELECT * FROM (
        SELECT x.*,ROW_NUMBER() OVER(PARTITION BY genai_ver_sid
          ORDER BY COALESCE(genai_ver_valid_end_fact_dttm,genai_ver_valid_start_fact_dttm,genai_ver_valid_crtn_dttm) DESC) rn
        FROM g_valid x
      ) z WHERE rn=1
    )
    SELECT COALESCE(x.genai_ver_valid_rslt_name,'NO_VALIDATION') d1,{gsig} d2,
           CAST(COUNT(*) AS DOUBLE) metric_value,COUNT(*) numerator,
           CAST(NULL AS BIGINT) denominator
    FROM gv v LEFT JOIN latest x ON v.genai_ver_sid=x.genai_ver_sid
    GROUP BY COALESCE(x.genai_ver_valid_rslt_name,'NO_VALIDATION'),{gsig}
""")

add_query(47, "Контроли валидации GenAI", "genai_validation", "GENAI", "risk_rate_pct_or_days",
          "indicator", "result", "t_genai_ver_valid", "Пробелы отчёта/альтернативной метрики и длительность процесса.", f"""
    WITH base AS (
      SELECT COALESCE(genai_ver_valid_rslt_name,'$NULL$') result,
             CASE WHEN {artifact_missing('genai_ver_valid_report_sid')} THEN 1 ELSE 0 END report_gap,
             CASE WHEN {truth('genai_ver_valid_alt_flag')} AND ({blank('genai_ver_valid_alt_metric_name')} OR {blank('genai_ver_valid_alt_metric_val')}) THEN 1 ELSE 0 END alt_gap,
             CASE WHEN {real_date('genai_ver_valid_start_fact_dttm')}
                        AND {real_date('genai_ver_valid_end_fact_dttm')}
                  THEN DATEDIFF(DATE(genai_ver_valid_end_fact_dttm),DATE(genai_ver_valid_start_fact_dttm)) END days
      FROM g_valid
    ), gaps AS (
      SELECT indicator d1,result d2,100.0*SUM(is_gap)/COUNT(*) metric_value,SUM(is_gap) numerator,COUNT(*) denominator
      FROM (SELECT result,STACK(2,'report_missing',report_gap,'alt_metric_missing',alt_gap) AS (indicator,is_gap) FROM base) x
      GROUP BY indicator,result
    ), duration AS (
      SELECT 'duration_p90_days' d1,result d2,PERCENTILE_APPROX(days,0.90) metric_value,
             CAST(NULL AS BIGINT) numerator,COUNT(days) denominator FROM base WHERE days>=0 GROUP BY result
    ) SELECT * FROM gaps UNION ALL SELECT * FROM duration
""")

add_query(48, "Мониторинг GenAI", "genai_monitoring", "GENAI", "risk_rate_pct",
          "indicator", "significance", "t_genai_ver", f"Пустые признаки отделены от явного отключения; контролируется результат старше {STALE_MONITORING_DAYS} дней.", f"""
    SELECT indicator d1,significance d2,100.0*SUM(is_gap)/COUNT(*) metric_value,SUM(is_gap) numerator,COUNT(*) denominator
    FROM (
      SELECT {gsig} significance,STACK(5,
        'model_monitoring_flag_missing',CASE WHEN {blank('genai_ver_montrg_auto_model_flag')} THEN 1 ELSE 0 END,
        'data_monitoring_flag_missing',CASE WHEN {blank('genai_ver_montrg_auto_data_flag')} THEN 1 ELSE 0 END,
        'model_exclusion_reason_missing',CASE WHEN {falsehood('genai_ver_montrg_auto_model_flag')} AND {blank('genai_ver_montrg_auto_exclude_reason_model_array')} THEN 1 ELSE 0 END,
        'data_exclusion_reason_missing',CASE WHEN {falsehood('genai_ver_montrg_auto_data_flag')} AND {blank('genai_ver_montrg_auto_exclude_reason_data_array')} THEN 1 ELSE 0 END,
        'last_result_missing_or_stale',CASE WHEN NOT {real_date('genai_ver_montrg_last_rslt_dttm')} OR DATEDIFF(CURRENT_DATE(),DATE(genai_ver_montrg_last_rslt_dttm))>{STALE_MONITORING_DAYS} THEN 1 ELSE 0 END
      ) AS (indicator,is_gap) FROM gv
    ) x GROUP BY indicator,significance
""")

add_query(49, "Ключевые метрики GenAI", "genai_metrics", "GENAI", "risk_rate_pct",
          "indicator", "significance", "t_genai_ver", "Проверяет наличие названия, значения и автоматической оценки ключевой метрики.", f"""
    SELECT indicator d1,significance d2,100.0*SUM(is_gap)/COUNT(*) metric_value,SUM(is_gap) numerator,COUNT(*) denominator
    FROM (
      SELECT {gsig} significance,STACK(3,
        'metric_code_missing',CASE WHEN {blank('genai_ver_metric_key_code')} THEN 1 ELSE 0 END,
        'metric_value_missing',CASE WHEN {blank('genai_ver_metric_key_val')} THEN 1 ELSE 0 END,
        'auto_assessment_missing',CASE WHEN {blank('genai_ver_metric_assmnt_auto_cmnt_txt')} THEN 1 ELSE 0 END
      ) AS (indicator,is_gap) FROM gv
    ) x GROUP BY indicator,significance
""")

add_query(50, "Экономика и лимиты использования GenAI", "genai_usage", "GENAI", "risk_rate_pct",
          "indicator", "significance", "t_genai_ver", "Эффект оценивается только среди версий с положительным планом; полнота лимитов — по всему портфелю.", f"""
    WITH base AS (
      SELECT {gsig} significance,
             CAST(genai_ver_fin_effect_plan_million_qty AS DOUBLE) plan_value,
             CAST(genai_ver_fin_effect_fact_million_qty AS DOUBLE) fact_value,
             genai_ver_usg_days_qty usage_days,
             genai_ver_usg_same_time_cnt concurrent_usage
      FROM gv
    ), metrics AS (
      SELECT significance,'financial_plan_missing' indicator,
             SUM(CASE WHEN plan_value IS NULL THEN 1 ELSE 0 END) bad,COUNT(*) total
      FROM base GROUP BY significance
      UNION ALL
      SELECT significance,'financial_fact_missing_or_below_plan',
             SUM(CASE WHEN fact_value IS NULL OR fact_value<plan_value THEN 1 ELSE 0 END),COUNT(*)
      FROM base WHERE plan_value>0 GROUP BY significance
      UNION ALL
      SELECT significance,'usage_days_limit_missing',
             SUM(CASE WHEN usage_days IS NULL THEN 1 ELSE 0 END),COUNT(*)
      FROM base GROUP BY significance
      UNION ALL
      SELECT significance,'concurrent_usage_limit_missing',
             SUM(CASE WHEN concurrent_usage IS NULL THEN 1 ELSE 0 END),COUNT(*)
      FROM base GROUP BY significance
    )
    SELECT indicator d1,significance d2,
           CASE WHEN total=0 THEN CAST(NULL AS DOUBLE) ELSE 100.0*bad/total END metric_value,
           bad numerator,total denominator
    FROM metrics
""")


if len(results) != 50:
    raise RuntimeError(f"Ожидалось 50 запросов, сформировано: {len(results)}")

final_df = reduce(lambda left, right: left.unionByName(right), results).persist(StorageLevel.MEMORY_AND_DISK)

# Удобный предпросмотр перед записью. Это агрегаты, поэтому объём ограничен.
final_df.orderBy("query_id", F.desc("metric_value")).show(SHOW_ROWS, truncate=False)

(
    final_df
    .repartition(1)
    .write
    .mode("overwrite")
    .format("parquet")
    .saveAsTable(OUT_TABLE)
)

print(f"Готово: {OUT_TABLE}")

final_df.unpersist()
for frame in cached_frames:
    frame.unpersist()
