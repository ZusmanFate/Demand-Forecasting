from flows.orchestrator import ForecastOrchestrator
import structlog

if __name__ == "__main__":
    logger = structlog.get_logger()
    try:
        orchestrator = ForecastOrchestrator()
        # 手动调用步骤1和步骤2（可通过修改源码暴露单步方法，或直接执行全流程后终止）
        logger.info("🚀 测试：原始数据导出+特征工程")
        # 执行全流程，到特征工程完成后手动终止（或注释后续步骤）
        orchestrator.run_full_pipeline()
    except Exception as e:
        logger.error(f"❌ 测试失败：{e}")