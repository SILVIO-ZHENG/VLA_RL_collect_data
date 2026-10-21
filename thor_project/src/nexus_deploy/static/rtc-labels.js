// RTC labels follow the selected checkpoint's server route, not its display name.
export function modelVersionForConfig(bootstrap, config = {}, rtcModelModes = {}) {
  const model = (bootstrap?.models || []).find((item) => String(item.id) === String(config.model_id));
  const variant = model?.effector_variants?.[config.effector_profile];
  const version = (variant?.versions || []).find((item) => String(item.id) === String(config.model_version));
  if (!version) return null;
  return {
    ...version,
    rtc_inference_mode: version.rtc_inference_mode
      ?? rtcModelModes[config.model_id]?.[config.effector_profile]?.[config.model_version],
  };
}

export function rtcInferenceMode(config = {}, version) {
  const explicit = config.rtc_inference_mode;
  if (explicit === "trained" || explicit === "guided") return explicit;
  const checkpointDefault = version?.rtc_inference_mode_default ?? version?.rtc_inference_mode;
  return checkpointDefault === "trained" ? "trained" : "guided";
}

export function trainedRtcSupported(version) {
  if (typeof version?.trained_rtc_supported === "boolean") return version.trained_rtc_supported;
  if (Array.isArray(version?.rtc_modes)) return version.rtc_modes.includes("trained");
  return (version?.rtc_inference_mode_default ?? version?.rtc_inference_mode) === "trained";
}

export function inferenceSelectionValue(config = {}, version) {
  const mode = config.inference_mode || "rtc";
  return mode === "rtc" ? `rtc-${rtcInferenceMode(config, version)}` : mode;
}

export function parseInferenceSelection(value) {
  if (value === "rtc-trained" || value === "rtc-guided") {
    return { inference_mode: "rtc", rtc_inference_mode: value === "rtc-trained" ? "trained" : "guided" };
  }
  return { inference_mode: value || "rtc" };
}

export function inferenceModeLabel(mode, version, selectedRtcMode) {
  if (mode === "rtc-trained") return "Trained RTC";
  if (mode === "rtc-guided") return "Inference RTC";
  if (mode === "rtc") {
    if (selectedRtcMode === "trained") return "Trained RTC";
    if (selectedRtcMode === "guided") return "Inference RTC";
    if (version?.rtc_inference_mode_default === "trained") return "Trained RTC";
    if (version?.rtc_inference_mode_default === "guided") return "Inference RTC";
    if (version?.rtc_inference_mode === "trained") return "Trained RTC";
    if (version?.rtc_inference_mode === "guided") return "Inference RTC";
    return "RTC";
  }
  return { sync: "同步", async: "异步" }[mode] || mode;
}

export function modelVersionLabel(bootstrap, config = {}) {
  const version = modelVersionForConfig(bootstrap, config);
  return version?.label || version?.id || config.model_version || "—";
}
