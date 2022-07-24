import React from "react";
import * as assert from "assert";
import {useHistory, useLocation, useParams, withRouter} from "react-router-dom";

const UrlHelper = React.createContext(null);

class UrlHelperWrapper extends React.Component {
  componentDidMount() {
  }

  componentDidUpdate(prevProps, prevState, snapshot) {
    super.componentDidUpdate(prevProps, prevState, snapshot);
  }

  componentWillUnmount() {
  }

  getParams() {
    return this.props.params;
  }

  setParams(params, allParamsKeys) {
    let url = new URL(this.props.history.location.search);

    for (let key in allParamsKeys) {
      if (params[key] !== undefined && params[key] != null) {
        url.searchParams.set(key, params[key]);
      } else {
        url.searchParams.delete(key);
      }
    }

    this.props.history.push(url.toString());
  }

  render() {
    return <UrlHelper.Provider value={{}}>
      {this.props.children}
    </UrlHelper.Provider>
  }
}

function UrlHelperWrapperWithParams (props) {
  return <UrlHelperWrapper {...props} history={useHistory()}/>
}

export {UrlHelper, UrlHelperWrapperWithParams as UrlHelperWrapper};